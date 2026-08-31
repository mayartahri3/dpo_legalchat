"""
export_for_colab.py — à exécuter EN LOCAL pour préparer feedback_export.zip,
le fichier à uploader dans Google Colab pour lancer colab_train.py (§7 —
même cycle question -> feedback -> correction -> validation -> dataset ->
entraînement, mais l'étape entraînement tourne sur GPU Colab au lieu du
poste local).

Usage :
    python export_for_colab.py
    python export_for_colab.py --include-reward-model
        # requis pour lancer un cycle PPO dans Colab SI le reward model
        # n'est pas ré-entraîné dans ce même notebook (voir colab_train.py
        # --method reward_model, qui évite d'avoir besoin de cette option).

Ce que fait ce script :
  1. Copie feedback.db (l'historique réel feedback/corrections/validations)
     et articles/ (le corpus juridique — nécessaire pour reconstruire
     exactement le même contexte RAG dans les prompts d'entraînement,
     voir dataset_builder.py) dans une archive zip.
  2. Optionnellement, inclut reward_models/ (déjà entraîné en local ou lors
     d'un précédent cycle Colab) — seul PPO en a besoin.
  3. N'inclut PAS models/ : on ne renvoie jamais un modèle déjà entraîné
     vers Colab, seulement les données sources.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="feedback.db")
    parser.add_argument("--data", default="articles")
    parser.add_argument("--reward-models-root", default="reward_models")
    parser.add_argument(
        "--include-reward-model",
        action="store_true",
        help="Inclut reward_models/ dans l'export — requis pour un cycle PPO dans Colab "
             "si le reward model n'est PAS entraîné dans ce même notebook Colab.",
    )
    parser.add_argument("--out", default="feedback_export.zip")
    args = parser.parse_args()

    db_path = Path(args.db)
    data_dir = Path(args.data)
    if not db_path.exists():
        raise FileNotFoundError(f"'{db_path}' introuvable — lancez ce script depuis le dossier backend/.")
    if not data_dir.exists():
        raise FileNotFoundError(f"'{data_dir}' introuvable — lancez ce script depuis le dossier backend/.")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        shutil.copy2(db_path, tmp / "feedback.db")
        shutil.copytree(data_dir, tmp / "articles")

        if args.include_reward_model:
            reward_root = Path(args.reward_models_root)
            if not reward_root.exists():
                raise FileNotFoundError(
                    f"'{reward_root}' introuvable (--include-reward-model demandé) — "
                    "entraînez d'abord un reward model en local, ou omettez cette option "
                    "et utilisez `colab_train.py --method reward_model` directement dans Colab."
                )
            shutil.copytree(reward_root, tmp / "reward_models")

        out_base = Path(args.out).with_suffix("")
        archive_path = shutil.make_archive(str(out_base), "zip", root_dir=str(tmp))

    print(f"Export prêt : {archive_path}")
    print("Uploadez ce fichier dans Colab, puis lancez :")
    print("    python colab_train.py --export-zip feedback_export.zip --method dpo   # ou ppo / grpo / reward_model")


if __name__ == "__main__":
    main()
