"""
colab_train.py — à exécuter DANS Google Colab (GPU) pour lancer un cycle
d'entraînement DPO, PPO, GRPO, ou reward_model, puis récupérer le résultat
en local via import_colab_model.py. Une seule méthode par run, comme
POST /api/train (--method dpo|ppo|grpo|reward_model).

Flux complet (§7, exécuté à distance) :
  1. EN LOCAL   : python export_for_colab.py  ->  feedback_export.zip
  2. DANS COLAB : uploader feedback_export.zip, puis
         !python colab_train.py --export-zip feedback_export.zip --method dpo
     (ou --method ppo / grpo / reward_model)
  3. Le script télécharge automatiquement :
         - trained_model_export.zip        (méthodes dpo / ppo / grpo)
         - reward_model_export.zip         (méthode reward_model)
  4. EN LOCAL   : python import_colab_model.py --export-dir <dossier dézippé>
     (détecte automatiquement lequel des deux registries mettre à jour)

Pourquoi PPO est différent : PPOTrainingPipeline (ppo_train.py) exige un
reward model déjà entraîné (RewardModelRegistry.latest()) — impossible de
lancer --method ppo sans lui. Deux options :
  (a) lancez d'abord --method reward_model DANS CE MÊME notebook Colab (le
      reward model reste dans reward_models/ pour le run suivant, pas
      besoin de ré-upload) ;
  (b) ré-exportez en local un reward model déjà entraîné avec
      `export_for_colab.py --include-reward-model` avant l'upload.

Cellule Colab typique avant de lancer ce script :
    !pip install -q -U trl transformers peft datasets accelerate \
        sentence-transformers faiss-cpu
    from google.colab import files
    uploaded = files.upload()   # sélectionner feedback_export.zip

Ce script réutilise tel quel dpo_train.py / ppo_train.py / grpo_train.py /
reward_model.py / minirag.py / feedback_store.py — uploadez-les aussi dans
le même dossier Colab (ou clonez le repo).
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from dpo_train import DPOTrainingPipeline, ModelRegistry
from ppo_train import PPOTrainingPipeline
from grpo_train import GRPOTrainingPipeline
from reward_model import RewardModelPipeline, RewardModelRegistry
from feedback_store import FeedbackStore
from minirag import EmbeddingIndex, load_articles


WORKDIR = Path("colab_workspace")


def _unpack_export(export_zip: str | Path) -> None:
    """Dézippe feedback_export.zip (produit par export_for_colab.py) dans WORKDIR."""
    WORKDIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(export_zip) as zf:
        zf.extractall(WORKDIR)
    if not (WORKDIR / "feedback.db").exists():
        raise FileNotFoundError(
            f"'feedback.db' introuvable dans {export_zip} — régénérez l'export avec "
            "export_for_colab.py côté local."
        )
    if not (WORKDIR / "articles").exists():
        raise FileNotFoundError(f"'articles/' introuvable dans {export_zip}.")


def _zip_dir(src_dir: Path, out_name: str) -> Path:
    """Zippe src_dir (ex: colab_workspace/models) de sorte que le zip contienne
    directement 'models/registry.json', 'models/v3/...' à sa racine — le
    format attendu par import_colab_model.py."""
    out_base = Path(out_name).with_suffix("")
    archive = shutil.make_archive(str(out_base), "zip", root_dir=str(src_dir.parent), base_dir=src_dir.name)
    return Path(archive)


def _try_colab_download(path: Path) -> None:
    """Déclenche le téléchargement navigateur si on tourne dans Colab ; sinon
    laisse simplement le zip sur disque (ex. si vous testez ce script en local)."""
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
        help="Cap à quelques steps/séquences courtes — pour valider la mécanique du pipeline "
             "avant de lancer un vrai run GPU complet (utile pour PPO/GRPO, plus lents que DPO).",
    )
    args = parser.parse_args()

    _unpack_export(args.export_zip)

    print(f"Chargement des articles depuis {WORKDIR / 'articles'} ...")
    articles = load_articles(WORKDIR / "articles")
    index = EmbeddingIndex(args.embed_model)
    index.build(articles, cache_dir=WORKDIR / "embedding_cache")

    store = FeedbackStore(WORKDIR / "feedback.db")

    # Registries construits DANS WORKDIR — ne mélange pas avec un éventuel
    # models/ ou reward_models/ local si ce script est testé hors Colab.
    models_root = WORKDIR / "models"
    reward_root = WORKDIR / "reward_models"

    try:
        if args.method == "reward_model":
            pipeline = RewardModelPipeline(base_model=args.llm_model, registry=RewardModelRegistry(reward_root))
            rv = pipeline.run_cycle(
                store, index, min_examples=args.min_examples, epochs=args.epochs, fast_test=args.fast_test
            )
            print(f"Reward model entraîné : v{rv.version} -> {rv.path}")
            out_zip = _zip_dir(reward_root, "reward_model_export")

        else:
            registry = ModelRegistry(models_root)
            if args.method == "dpo":
                pipeline = DPOTrainingPipeline(base_model=args.llm_model, registry=registry)
            elif args.method == "ppo":
                reward_registry = RewardModelRegistry(reward_root)
                if reward_registry.latest() is None:
                    raise ValueError(
                        "Aucun reward model trouvé dans reward_models/ — lancez d'abord "
                        "`--method reward_model` dans ce même notebook, ou ré-exportez en local "
                        "avec `export_for_colab.py --include-reward-model` avant l'upload."
                    )
                pipeline = PPOTrainingPipeline(
                    base_model=args.llm_model, registry=registry, reward_registry=reward_registry
                )
            else:  # grpo
                pipeline = GRPOTrainingPipeline(base_model=args.llm_model, registry=registry)

            mv = pipeline.run_cycle(
                store, index, min_examples=args.min_examples, epochs=args.epochs, fast_test=args.fast_test
            )
            print(f"Version entraînée : v{mv.version} (méthode={mv.method}) -> {mv.path}")
            out_zip = _zip_dir(models_root, "trained_model_export")
    finally:
        store.close()

    print(f"\nExport prêt : {out_zip}")
    _try_colab_download(out_zip)
    print(
        "\nÉtape suivante (en local) :\n"
        f"    unzip {out_zip.name}\n"
        f"    python import_colab_model.py --export-dir {out_zip.stem}"
    )


if __name__ == "__main__":
    main()
