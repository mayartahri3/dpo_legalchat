"""
import_colab_model.py — importe un export Colab (trained_model_export/ ou
reward_model_export/) dans les registries locaux.

    python import_colab_model.py --export-dir ./trained_model_export

Les versions sont renumérotées selon VOTRE registry local, importées en status="staged",
et le `method` (dpo/ppo/grpo) est préservé.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from reward_model import RewardModelRegistry, RewardModelVersion
from train_common import ModelRegistry, ModelVersion


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _import_all(export_dir: Path, sub: str, local_root: str, registry, build_entry) -> list[int]:
    entries = json.loads((export_dir / sub / "registry.json").read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"{sub}/registry.json est vide — rien à importer.")

    imported = []
    for e in entries:
        src = export_dir / e["path"]
        if not src.exists():
            print(f"⚠ Poids introuvables pour v{e['version']} ({src}), ignoré.")
            continue
        version = registry.next_version()
        dst = Path(local_root) / f"v{version}"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        registry.register(build_entry(version, e, str(dst)))
        imported.append(version)
        print(f"✓ Importé : v{version} <- {src} ({e['n_training_pairs']} paires)")
    return imported


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--reward-models-root", default="reward_models")
    args = ap.parse_args()

    export_dir = Path(args.export_dir)
    has_models = (export_dir / "models" / "registry.json").exists()
    has_reward = (export_dir / "reward_models" / "registry.json").exists()
    if not (has_models or has_reward):
        raise FileNotFoundError(f"Ni models/registry.json ni reward_models/registry.json dans {export_dir}.")

    models, rewards = [], []
    if has_models:
        models = _import_all(
            export_dir, "models", args.models_root, ModelRegistry(args.models_root),
            lambda v, e, p: ModelVersion(
                version=v, base_model=e["base_model"], created_at=e.get("created_at", _now()),
                dataset_version=e.get("dataset_version", 0), n_training_pairs=e["n_training_pairs"],
                path=p, status="staged", method=e.get("method", "dpo")),
        )
    if has_reward:
        rewards = _import_all(
            export_dir, "reward_models", args.reward_models_root,
            RewardModelRegistry(args.reward_models_root),
            lambda v, e, p: RewardModelVersion(
                version=v, base_model=e["base_model"], created_at=e.get("created_at", _now()),
                n_training_pairs=e["n_training_pairs"], path=p),
        )

    if models:
        print(f"\nVersions de modèle créées : {models}")
        print("Étape suivante : `python eval_metrics.py --split test`, puis « Promouvoir » dans l'UI.")
    if rewards:
        print(f"Versions de reward model créées : {rewards} (utilisables directement par PPO).")


if __name__ == "__main__":
    main()