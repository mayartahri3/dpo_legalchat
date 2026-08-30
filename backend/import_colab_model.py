"""
import_colab_model.py — à exécuter EN LOCAL après avoir téléchargé et
dézippé trained_model_export.zip (issu de colab_train.py).

Usage :
    python import_colab_model.py --export-dir ./trained_model_export

Ce script :
  1. Copie le dossier de poids (models/vN/) dans votre répertoire models/ local.
  2. Réattribue un numéro de version cohérent avec VOTRE registry.json local
     (celui de Colab peut entrer en collision avec des versions déjà
     entraînées localement entre-temps).
  3. Enregistre l'entrée dans models/registry.json avec status="staged" —
     à évaluer/promouvoir comme n'importe quel autre cycle (§7-étape 8).
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dpo_train import ModelRegistry, ModelVersion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True,
                         help="Dossier dézippé de l'export Colab (contient models/ et datasets/)")
    parser.add_argument("--models-root", default="models")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    exported_registry_path = export_dir / "models" / "registry.json"
    if not exported_registry_path.exists():
        raise FileNotFoundError(f"registry.json introuvable dans {exported_registry_path}")

    exported_entries = json.loads(exported_registry_path.read_text(encoding="utf-8"))
    if not exported_entries:
        raise ValueError("Le registry exporté est vide — rien à importer.")

    # On importe la/les entrée(s) 'staged' les plus récentes (pas déjà connues localement).
    local_registry = ModelRegistry(root=args.models_root)
    local_known_paths = {e["path"] for e in local_registry.history()}

    imported = []
    for entry in exported_entries:
        exported_weights_dir = export_dir / entry["path"]  # ex: export_dir/models/v3
        if not exported_weights_dir.exists():
            print(f"⚠ Poids introuvables pour v{entry['version']} ({exported_weights_dir}), ignoré.")
            continue

        new_version = local_registry.next_version()
        local_weights_dir = Path(args.models_root) / f"v{new_version}"
        if str(local_weights_dir) in local_known_paths:
            print(f"⚠ v{new_version} déjà présent en local, ignoré (déjà importé ?).")
            continue

        shutil.copytree(exported_weights_dir, local_weights_dir, dirs_exist_ok=True)

        mv = ModelVersion(
            version=new_version,
            base_model=entry["base_model"],
            created_at=entry.get("created_at", datetime.now(timezone.utc).isoformat()),
            dataset_version=entry["dataset_version"],
            n_training_pairs=entry["n_training_pairs"],
            path=str(local_weights_dir),
            status="staged",   # toujours évaluer avant promote, même si entraîné sur GPU
        )
        local_registry.register(mv)
        imported.append(new_version)
        print(f"✓ Importé : v{new_version} <- {exported_weights_dir} ({mv.n_training_pairs} paires)")

    if not imported:
        print("Rien d'importé.")
    else:
        print(f"\nTerminé. Versions locales créées : {imported}")
        print("Étape suivante : évaluez (ReferenceRewardScorer / revue manuelle) "
              "puis `python legal_rag_app.py promote <version>` pour mettre en production.")


if __name__ == "__main__":
    main()
