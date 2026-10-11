"""
export_for_colab.py — prépare feedback_export.zip à uploader dans Colab.

    python export_for_colab.py [--include-reward-model]

Contient : feedback.db, articles/, eval_split.json (s'il existe — INDISPENSABLE pour
que Colab n'entraîne pas sur les questions de test), et reward_models/ si demandé
(seul PPO en a besoin). Jamais models/.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="feedback.db")
    ap.add_argument("--data", default="articles")
    ap.add_argument("--reward-models-root", default="reward_models")
    ap.add_argument("--include-reward-model", action="store_true")
    ap.add_argument("--out", default="feedback_export.zip")
    args = ap.parse_args()

    required = {"--db": Path(args.db), "--data": Path(args.data)}
    if args.include_reward_model:
        required["--reward-models-root"] = Path(args.reward_models_root)
    for flag, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"'{p}' introuvable ({flag}) — lancez depuis le dossier backend/.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy2(args.db, tmp / "feedback.db")
        shutil.copytree(args.data, tmp / "articles")
        if Path("eval_split.json").exists():
            shutil.copy2("eval_split.json", tmp / "eval_split.json")
        else:
            print("⚠ eval_split.json absent : Colab entraînera sur TOUTES les corrections "
                  "(pas de holdout). Créez-le avec `python eval_metrics.py --make-split`.")
        if args.include_reward_model:
            shutil.copytree(args.reward_models_root, tmp / "reward_models")
        archive = shutil.make_archive(str(Path(args.out).with_suffix("")), "zip", root_dir=str(tmp))

    print(f"Export prêt : {archive}")
    print("Dans Colab : python colab_train.py --export-zip feedback_export.zip --method dpo   # ou ppo / grpo / reward_model")


if __name__ == "__main__":
    main()