"""
import_colab_model.py — à exécuter EN LOCAL après avoir téléchargé et
dézippé l'export produit par colab_train.py :
    - trained_model_export.zip  pour un cycle dpo / ppo / grpo
    - reward_model_export.zip   pour un cycle reward_model

Usage :
    python import_colab_model.py --export-dir ./trained_model_export
    python import_colab_model.py --export-dir ./reward_model_export

Ce script :
  1. Détecte automatiquement le type d'export — models/registry.json
     (modèle génératif) et/ou reward_models/registry.json (reward model) —
     et importe dans le(s) registry(ies) local(aux) correspondant(s).
  2. Copie le(s) dossier(s) de poids dans votre répertoire local (models/
     et/ou reward_models/).
  3. Réattribue un numéro de version cohérent avec VOTRE registry local
     (celui de Colab peut entrer en collision avec des versions déjà
     entraînées localement entre-temps).
  4. Enregistre l'entrée avec status="staged" pour un modèle génératif —
     à évaluer/comparer (page Entraînement DPO → cocher les versions →
     Comparer) puis promouvoir comme n'importe quel autre cycle. Les
     reward models n'ont pas de notion "staged/active" (reward_model.py).
  5. Préserve le `method` (dpo/ppo/grpo) de l'entrée exportée. C'est le
     correctif clé pour PPO/GRPO : sans lui, toute version importée
     retombait sur method="dpo" par défaut et faussait l'étiquette
     affichée dans la comparaison de versions (Training.jsx).
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dpo_train import ModelRegistry, ModelVersion
from reward_model import RewardModelRegistry, RewardModelVersion


def _import_generative_models(export_dir: Path, models_root: str) -> list[int]:
    exported_registry_path = export_dir / "models" / "registry.json"
    exported_entries = json.loads(exported_registry_path.read_text(encoding="utf-8"))
    if not exported_entries:
        raise ValueError("Le registry exporté (models/registry.json) est vide — rien à importer.")

    local_registry = ModelRegistry(root=models_root)
    local_known_paths = {e["path"] for e in local_registry.history()}

    imported = []
    for entry in exported_entries:
        exported_weights_dir = export_dir / entry["path"]  # ex: export_dir/models/v3
        if not exported_weights_dir.exists():
            print(f"⚠ Poids introuvables pour v{entry['version']} ({exported_weights_dir}), ignoré.")
            continue

        new_version = local_registry.next_version()
        local_weights_dir = Path(models_root) / f"v{new_version}"
        if str(local_weights_dir) in local_known_paths:
            print(f"⚠ v{new_version} déjà présent en local, ignoré (déjà importé ?).")
            continue

        shutil.copytree(exported_weights_dir, local_weights_dir, dirs_exist_ok=True)

        mv = ModelVersion(
            version=new_version,
            base_model=entry["base_model"],
            created_at=entry.get("created_at", datetime.now(timezone.utc).isoformat()),
            dataset_version=entry.get("dataset_version", 0),
            n_training_pairs=entry["n_training_pairs"],
            path=str(local_weights_dir),
            status="staged",             # toujours évaluer avant promote, même si entraîné sur GPU
            method=entry.get("method", "dpo"),  # préserve dpo/ppo/grpo — voir docstring du module
        )
        local_registry.register(mv)
        imported.append(new_version)
        print(f"✓ Importé : v{new_version} [{mv.method}] <- {exported_weights_dir} ({mv.n_training_pairs} paires)")
    return imported


def _import_reward_model(export_dir: Path, reward_models_root: str) -> list[int]:
    exported_registry_path = export_dir / "reward_models" / "registry.json"
    exported_entries = json.loads(exported_registry_path.read_text(encoding="utf-8"))
    if not exported_entries:
        raise ValueError("Le registry exporté (reward_models/registry.json) est vide — rien à importer.")

    local_registry = RewardModelRegistry(root=reward_models_root)
    local_known_paths = {e["path"] for e in local_registry.history()}

    imported = []
    for entry in exported_entries:
        exported_weights_dir = export_dir / entry["path"]
        if not exported_weights_dir.exists():
            print(f"⚠ Poids introuvables pour v{entry['version']} ({exported_weights_dir}), ignoré.")
            continue

        new_version = local_registry.next_version()
        local_weights_dir = Path(reward_models_root) / f"v{new_version}"
        if str(local_weights_dir) in local_known_paths:
            print(f"⚠ v{new_version} déjà présent en local, ignoré (déjà importé ?).")
            continue

        shutil.copytree(exported_weights_dir, local_weights_dir, dirs_exist_ok=True)

        rv = RewardModelVersion(
            version=new_version,
            base_model=entry["base_model"],
            created_at=entry.get("created_at", datetime.now(timezone.utc).isoformat()),
            n_training_pairs=entry["n_training_pairs"],
            path=str(local_weights_dir),
        )
        local_registry.register(rv)
        imported.append(new_version)
        print(f"✓ Importé (reward model) : v{new_version} <- {exported_weights_dir} ({rv.n_training_pairs} paires)")
    return imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True,
                         help="Dossier dézippé de l'export Colab (trained_model_export/ ou reward_model_export/)")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--reward-models-root", default="reward_models")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    has_generative = (export_dir / "models" / "registry.json").exists()
    has_reward = (export_dir / "reward_models" / "registry.json").exists()

    if not has_generative and not has_reward:
        raise FileNotFoundError(
            f"Ni models/registry.json ni reward_models/registry.json trouvés dans {export_dir} — "
            "vérifiez que le dossier a bien été dézippé depuis un export produit par colab_train.py."
        )

    imported_models: list[int] = []
    imported_rewards: list[int] = []
    if has_generative:
        imported_models = _import_generative_models(export_dir, args.models_root)
    if has_reward:
        imported_rewards = _import_reward_model(export_dir, args.reward_models_root)

    if not imported_models and not imported_rewards:
        print("Rien d'importé.")
        return

    print("\nTerminé.")
    if imported_models:
        print(f"Versions de modèle locales créées : {imported_models}")
        print("Étape suivante : comparez-les (page Entraînement DPO → cocher les versions → "
              "Comparer) puis promouvez la meilleure avec "
              "`python legal_rag_app.py promote <version>` ou le bouton « Promouvoir ».")
    if imported_rewards:
        print(f"Versions de reward model locales créées : {imported_rewards}")
        print("Utilisable immédiatement par un cycle PPO local (POST /api/train method=ppo) "
              "sans étape supplémentaire.")


if __name__ == "__main__":
    main()