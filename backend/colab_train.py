"""
colab_train.py — à lancer DANS Google Colab (GPU), depuis le dossier qui contient
les .py du backend. Entraîne DPO, GRPO et reward model, évalue tout sur le
holdout (eval_metrics.py), puis produit trained_model_export.zip pour
import_colab_model.py.

    python colab_train.py --export-zip feedback_export.zip --method dpo --fast-test   # test rapide
    python colab_train.py --export-zip feedback_export.zip --method all --epochs 3    # run complet

Choix de conception :
  - Chaque méthode tourne dans son PROPRE process : la mémoire GPU est entièrement
    libérée entre deux méthodes (plus besoin de redémarrer le runtime Colab).
  - Le script se place toujours dans son propre dossier : dataset_builder.py cherche
    eval_split.json à côté de lui, et models/ est un chemin relatif. Ainsi le holdout
    est bien appliqué et les poids atterrissent là où registry.json les attend.
  - Un échec (ex. OOM) n'arrête pas les autres méthodes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["dpo", "grpo", "reward_model"]   # PPO retiré : absent des versions récentes de trl
EXPORT_DIR = "trained_model_export"


# --------------------------------------------------------------------------- #
# Worker : UNE méthode, dans un process dédié
# --------------------------------------------------------------------------- #

def worker(args) -> None:
    from feedback_store import FeedbackStore
    from minirag import EmbeddingIndex, load_articles
    from train_common import free_memory

    index = EmbeddingIndex(args.embed_model)
    index.build(load_articles("articles"), cache_dir="embedding_cache")
    store = FeedbackStore("feedback.db")
    common = dict(min_examples=args.min_examples, epochs=args.epochs, fast_test=args.fast_test)
    m = args.method

    try:
        if m == "dpo":
            from dpo_train import DPOTrainingPipeline
            r = DPOTrainingPipeline(base_model=args.base_model).run_cycle(
                store, index, gradient_accumulation_steps=args.grad_accum, **common)
        elif m == "grpo":
            from grpo_train import GRPOTrainingPipeline
            r = GRPOTrainingPipeline(base_model=args.base_model).run_cycle(
                store, index, num_generations=args.num_generations,
                max_completion_length=args.max_completion_length, **common)
        elif m == "reward_model":
            from reward_model import RewardModelPipeline
            r = RewardModelPipeline(base_model=args.base_model).run_cycle(store, index, **common)
        else:
            raise ValueError(f"méthode inconnue : {m}")
        print(f"\n[{m}] OK -> v{r.version} ({r.n_training_pairs} paires) -> {r.path}")
    finally:
        store.close()
        free_memory()


# --------------------------------------------------------------------------- #
# Orchestrateur
# --------------------------------------------------------------------------- #

def run_tee(cmd: list[str], out_file: Path) -> int:
    """Lance cmd, affiche la sortie en direct ET l'enregistre dans out_file."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    out_file.write_text("".join(lines), encoding="utf-8")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-zip", default="feedback_export.zip")
    ap.add_argument("--method", default="all", choices=["all", *ORDER])
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=3,
                    help="Identique pour toutes les méthodes (comparaison équitable).")
    ap.add_argument("--min-examples", type=int, default=4,
                    help="Minimum de paires d'entraînement (le défaut des pipelines, 20, bloque avec un petit dataset).")
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="DPO : plus petit = plus de pas d'optimisation avec peu de données.")
    ap.add_argument("--num-generations", type=int, default=2, help="Taille du groupe GRPO.")
    ap.add_argument("--max-completion-length", type=int, default=200, help="GRPO.")
    ap.add_argument("--fast-test", action="store_true", help="3 steps par méthode : vérifie que tout tourne.")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--keep-existing", action="store_true",
                    help="Ne pas effacer models/ et reward_models/ au démarrage "
                         "(pour AJOUTER une méthode à un run précédent).")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    zip_path = Path(args.export_zip).resolve()   # à résoudre AVANT le chdir
    os.chdir(HERE)
    sys.path.insert(0, str(HERE))

    if args.worker:
        worker(args)
        return

    # --- 0. GPU ---------------------------------------------------------- #
    if shutil.which("nvidia-smi") is None and not args.allow_cpu:
        sys.exit("Aucun GPU détecté. Colab : Runtime > Modifier le type d'exécution > GPU (T4). "
                 "(--allow-cpu pour forcer, déconseillé.)")

    # --- 1. Export local -> workspace ------------------------------------ #
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(HERE)
    for needed in ("feedback.db", "articles"):
        if not (HERE / needed).exists():
            sys.exit(f"'{needed}' introuvable dans {HERE} — uploadez feedback_export.zip "
                     "(généré par export_for_colab.py juste avant).")

    has_split = (HERE / "eval_split.json").exists()
    if not has_split:
        print("⚠ eval_split.json absent : entraînement ET évaluation sur les mêmes questions "
              "(mémorisation, pas de généralisation). Créez-le en local avec "
              "`python eval_metrics.py --make-split` avant d'exporter.")

    from feedback_store import FeedbackStore
    from train_common import count_usable_pairs
    store = FeedbackStore("feedback.db")
    n = count_usable_pairs(store)
    store.close()
    print(f"{n} paire(s) d'entraînement utilisable(s) (holdout exclu).")
    if n < 2:
        sys.exit("Pas assez de corrections validées pour entraîner.")
    args.min_examples = min(args.min_examples, n)

    methods = ORDER if args.method == "all" else [args.method]

    # --- 2. Run propre : un models/ hérité (ex. copié depuis votre PC, chemins Windows
    #        'models\v1') casserait l'évaluation et fausserait la numérotation --- #
    if "/content/drive" in str(HERE):
        print("⚠ Vous travaillez DANS Google Drive (lent, l'export peut sembler bloqué). "
              "Recommandé : !cp -r /content/drive/MyDrive/backend /content/backend && cd /content/backend")
    if not args.keep_existing:
        for d in ("models", "reward_models", EXPORT_DIR):
            shutil.rmtree(HERE / d, ignore_errors=True)
        for f in ("eval_report.json", "eval_table.txt"):
            (HERE / f).unlink(missing_ok=True)

    # --- 3. Entraînement : un process par méthode ------------------------ #
    passthrough = [
        "--base-model", args.base_model, "--embed-model", args.embed_model,
        "--epochs", str(args.epochs), "--min-examples", str(args.min_examples),
        "--grad-accum", str(args.grad_accum), "--num-generations", str(args.num_generations),
        "--max-completion-length", str(args.max_completion_length),
    ] + (["--fast-test"] if args.fast_test else [])
    status = {}
    for m in methods:
        print(f"\n{'=' * 64}\n  {m.upper()}\n{'=' * 64}", flush=True)
        rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--worker",
                              "--method", m, *passthrough])
        status[m] = "OK" if rc == 0 else f"ÉCHEC (code {rc}) — voir la trace ci-dessus"

    print("\nRésumé de l'entraînement :")
    for m, s in status.items():
        print(f"  {m:<13} {s}")

    if not any(s == "OK" for s in status.values()):
        sys.exit("\nAucune méthode n'a abouti : rien à évaluer ni à exporter. "
                 "Corrigez l'erreur de la trace ci-dessus (la plus ancienne), puis relancez.")

    # --- 4. Évaluation sur le holdout (base vs chaque version) ------------ #
    if not args.skip_eval and (HERE / "models" / "registry.json").exists():
        split = "test" if has_split else "all"
        cmd = [sys.executable, "eval_metrics.py", "--split", split, "--write-registry",
               "--out", "eval_report.json", "--base-model", args.base_model,
               "--embed-model", args.embed_model, "--max-new-tokens", "200"]
        if args.fast_test:
            cmd += ["--n-questions", "2"]
        print(f"\n{'=' * 64}\n  ÉVALUATION (split={split})\n{'=' * 64}", flush=True)
        if run_tee(cmd, HERE / "eval_table.txt") != 0:
            print("⚠ L'évaluation a échoué ; les modèles sont quand même exportés. "
                  "Relancez `python eval_metrics.py --split test` séparément.")

    # --- 5. Export ---------------------------------------------------------- #
    out = HERE / EXPORT_DIR
    out.mkdir(exist_ok=True)
    for d in ("models", "reward_models"):
        if (HERE / d).exists():
            shutil.copytree(HERE / d, out / d, dirs_exist_ok=True)
    for f in ("eval_report.json", "eval_table.txt", "eval_split.json"):
        if (HERE / f).exists():
            shutil.copy2(HERE / f, out / f)
    archive = shutil.make_archive(str(HERE / EXPORT_DIR), "zip", root_dir=str(out))

    versions = []
    for d in ("models", "reward_models"):
        reg = HERE / d / "registry.json"
        if reg.exists():
            versions += [(d, e["version"], e.get("method", "reward_model"), e["n_training_pairs"])
                         for e in json.loads(reg.read_text(encoding="utf-8"))]
    print(f"\nExport prêt : {archive}")
    for d, v, meth, pairs in versions:
        print(f"  {d}/v{v}  {meth}  ({pairs} paires)")

    try:
        from google.colab import files  # type: ignore
        files.download(archive)
    except Exception:  # ImportError hors Colab ; AttributeError via `!python` (pas de kernel IPython)
        print(f"Téléchargement : dans une cellule, lancez\n"
              f"    from google.colab import files; files.download('{Path(archive).name}')\n"
              f"ou copiez-le dans Drive :  !cp {Path(archive).name} /content/drive/MyDrive/")
    print(f"\nEn local :\n    unzip {Path(archive).name} -d {EXPORT_DIR}\n"
          f"    python import_colab_model.py --export-dir ./{EXPORT_DIR}")


if __name__ == "__main__":
    main()