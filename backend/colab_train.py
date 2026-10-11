"""
colab_train.py — à exécuter DANS Google Colab (GPU) pour lancer UN cycle
d'entraînement DPO ou GRPO (ou PPO / reward_model), puis récupérer le
résultat en local via import_colab_model.py.

Flux complet (§7, exécuté à distance) :
  1. EN LOCAL   : python export_for_colab.py  ->  feedback_export.zip
                  (exportez JUSTE AVANT d'uploader, voir discussion sur le
                  décalage 25 vs 20 corrections — un export périmé entraîne
                  sur moins de données que ce que vous avez validé.)
  2. DANS COLAB : uploader feedback_export.zip + colab_code.zip, puis
         !python colab_train.py --export-zip feedback_export.zip --method dpo
     (ou --method grpo)
  3. Le script télécharge automatiquement trained_model_export.zip
  4. EN LOCAL   : python import_colab_model.py --export-dir <dossier dézippé>

IMPORTANT — un cycle par runtime :
  Entraînez DPO, téléchargez le résultat, PUIS Runtime > Redémarrer la
  session avant de lancer GRPO. Même avec le nettoyage mémoire ajouté dans
  ce fichier, un process Python qui a déjà chargé/déchargé un modèle
  plusieurs fois accumule de la fragmentation d'allocateur CUDA — un
  restart complet est la façon la plus fiable d'éviter un OOM sur le 2e
  cycle, surtout sur le quota gratuit (T4, ~16 Go VRAM).

Ce script réutilise dpo_train.py / grpo_train.py / ppo_train.py /
reward_model.py / minirag.py / feedback_store.py — uploadez-les aussi dans
le même dossier Colab (voir colab_code.zip dans le guide).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from dpo_train import DPOTrainingPipeline, ModelRegistry, print_gpu_mem, free_memory
from grpo_train import GRPOTrainingPipeline


from feedback_store import FeedbackStore
from minirag import EmbeddingIndex, load_articles


WORKDIR = Path("colab_workspace")


def _check_gpu() -> None:
    import torch
    if not torch.cuda.is_available():
        print(
            "\n⚠ ATTENTION : aucun GPU détecté dans ce runtime Colab.\n"
            "  Runtime > Modifier le type d'exécution > Accélérateur matériel > GPU (T4).\n"
            "  L'entraînement va continuer sur CPU si vous ne l'interrompez pas maintenant,\n"
            "  mais GRPO en particulier (génération pendant l'entraînement) sera extrêmement\n"
            "  lent sur CPU — attendez-vous à des heures plutôt que des minutes.\n"
        )
        return
    name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[colab_train] GPU détecté : {name} ({total_mem:.1f} Go VRAM)")
    if total_mem < 14:
        print(
            "⚠ Moins de 14 Go de VRAM détectés — réduisez encore num_generations (GRPO) "
            "et/ou passez --fast-test pour valider la mécanique avant un run complet."
        )


def _unpack_export(export_zip: str | Path) -> None:
    WORKDIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(export_zip) as zf:
        zf.extractall(WORKDIR)
    if not (WORKDIR / "feedback.db").exists():
        raise FileNotFoundError(
            f"'feedback.db' introuvable dans {export_zip} — régénérez l'export avec "
            "export_for_colab.py côté local, JUSTE AVANT d'uploader."
        )
    if not (WORKDIR / "articles").exists():
        raise FileNotFoundError(f"'articles/' introuvable dans {export_zip}.")


def _zip_dir(src_dir: Path, out_name: str) -> Path:
    out_base = Path(out_name).with_suffix("")
    archive = shutil.make_archive(str(out_base), "zip", root_dir=str(src_dir.parent), base_dir=src_dir.name)
    return Path(archive)


def _try_colab_download(path: Path) -> None:
    try:
        from google.colab import files  # type: ignore
        files.download(str(path))
    except ImportError:
        print(f"(Pas dans Colab — récupérez le fichier manuellement : {path})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-zip", required=True, help="feedback_export.zip (voir export_for_colab.py)")
    parser.add_argument("--method", required=True, choices=["dpo", "ppo", "grpo", "reward_model"])
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--min-examples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--fast-test", action="store_true",
        help="Cap à quelques steps/séquences courtes — valide la mécanique du pipeline "
             "avant un run complet, sans attendre un vrai entraînement.",
    )
    # Overrides mémoire, réglables sans toucher au code si un OOM survient malgré tout.
    parser.add_argument("--batch-size", type=int, default=1, help="per_device_train_batch_size")
    parser.add_argument("--grad-accum", type=int, default=8, help="gradient_accumulation_steps (DPO/reward_model)")
    parser.add_argument("--num-generations", type=int, default=2, help="taille du groupe GRPO")
    parser.add_argument("--max-completion-length", type=int, default=200, help="longueur max générée (GRPO)")
    args = parser.parse_args()

    _check_gpu()
    _unpack_export(args.export_zip)

    print(f"Chargement des articles depuis {WORKDIR / 'articles'} ...")
    articles = load_articles(WORKDIR / "articles")
    index = EmbeddingIndex(args.embed_model)
    index.build(articles, cache_dir=WORKDIR / "embedding_cache")

    store = FeedbackStore(WORKDIR / "feedback.db")

    models_root = WORKDIR / "models"
    reward_root = WORKDIR / "reward_models"

    try:
        if args.method == "reward_model":
            
            rv = pipeline.run_cycle(
                store, index, min_examples=args.min_examples, epochs=args.epochs, fast_test=args.fast_test
            )
            print(f"Reward model entraîné : v{rv.version} -> {rv.path}")
            out_zip = _zip_dir(reward_root, "reward_model_export")

        elif args.method == "dpo":
            registry = ModelRegistry(models_root)
            pipeline = DPOTrainingPipeline(base_model=args.llm_model, registry=registry)
            mv = pipeline.run_cycle(
                store, index, min_examples=args.min_examples, epochs=args.epochs, fast_test=args.fast_test,
                per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            )
            print(f"Version entraînée : v{mv.version} (méthode={mv.method}) -> {mv.path}")
            out_zip = _zip_dir(models_root, "trained_model_export_dpo")

        elif args.method == "grpo":
            registry = ModelRegistry(models_root)
            pipeline = GRPOTrainingPipeline(base_model=args.llm_model, registry=registry)
            mv = pipeline.run_cycle(
                store, index, min_examples=args.min_examples, epochs=args.epochs, fast_test=args.fast_test,
                per_device_train_batch_size=args.batch_size,
                num_generations=args.num_generations, max_completion_length=args.max_completion_length,
            )
            print(f"Version entraînée : v{mv.version} (méthode={mv.method}) -> {mv.path}")
            out_zip = _zip_dir(models_root, "trained_model_export_grpo")

            mv = pipeline.run_cycle(
                store, index, min_examples=args.min_examples, epochs=args.epochs, fast_test=args.fast_test
            )
            print(f"Version entraînée : v{mv.version} (méthode={mv.method}) -> {mv.path}")
            out_zip = _zip_dir(models_root, "trained_model_export")
    except Exception:
        print("\n✗ Le cycle a échoué. Voir la trace ci-dessus.")
        print("  Si c'est un OOM : relancez avec --batch-size 1 --grad-accum 16 (DPO) ou "
              "--num-generations 1 --max-completion-length 128 (GRPO), et/ou --fast-test "
              "d'abord pour vérifier que le reste du pipeline fonctionne.")
        raise
    finally:
        store.close()
        free_memory()  # nettoie même en cas d'échec, avant un éventuel 2e essai dans ce même runtime

    print(f"\nExport prêt : {out_zip}")
    _try_colab_download(out_zip)
    print(
        "\nÉtape suivante (en local) :\n"
        f"    unzip {out_zip.name}\n"
        f"    python import_colab_model.py --export-dir {out_zip.stem}\n"
        "\nSi vous comptez entraîner l'AUTRE méthode maintenant (dpo <-> grpo), "
        "redémarrez le runtime Colab d'abord (Runtime > Redémarrer la session) "
        "avant de relancer ce script — voir la note en haut de ce fichier."
    )


if __name__ == "__main__":
    main()