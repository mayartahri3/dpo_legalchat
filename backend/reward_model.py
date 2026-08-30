"""
reward_model.py — Entraîne un modèle de récompense explicite (§ comparaison
DPO / PPO / GRPO).

Pourquoi ce fichier n'existait pas avant : DPO n'a pas besoin de reward model
séparé (voir la note en tête de dpo_train.py — la log-vraisemblance relative
policy/référence joue ce rôle implicitement). PPO, en revanche, EN A BESOIN :
il génère des réponses pendant l'entraînement et a besoin d'un scoreur
explicite pour savoir si chaque génération est bonne ou mauvaise.

GRPO peut fonctionner SANS ce reward model (voir grpo_train.py — on peut lui
donner une fonction de récompense Python directe, ex. similarité d'embedding
à la réponse validée). Ce fichier reste utile pour PPO, et en option pour
GRPO si vous préférez un scoreur appris plutôt qu'une heuristique.

Le reward model est une tête de classification (1 sortie = score scalaire)
sur le MÊME modèle de base que la politique, entraînée sur les mêmes paires
(chosen, rejected) que le DPO, via trl.RewardTrainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dataset_builder import build_preference_dataset, training_columns
from feedback_store import FeedbackStore
from minirag import EmbeddingIndex


@dataclass
class RewardModelVersion:
    version: int
    base_model: str
    created_at: str
    n_training_pairs: int
    path: str
    status: str = "ready"   # ready | retired


class RewardModelRegistry:
    """Historique séparé de ModelRegistry — le reward model n'est pas un modèle
    génératif, il ne se "promeut" pas en production, il sert d'outil pour PPO."""

    def __init__(self, root: str | Path = "reward_models"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps([]), encoding="utf-8")

    def _load(self) -> list[dict]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, entries: list[dict]) -> None:
        self.index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def next_version(self) -> int:
        entries = self._load()
        return (max((e["version"] for e in entries), default=0)) + 1

    def register(self, rv: RewardModelVersion) -> None:
        entries = self._load()
        entries.append(rv.__dict__)
        self._save(entries)

    def latest(self) -> dict | None:
        entries = self._load()
        return entries[-1] if entries else None

    def history(self) -> list[dict]:
        return self._load()


class RewardModelPipeline:
    """
    Entraîne un reward model via trl.RewardTrainer sur les mêmes corrections
    validées que le DPO. Nécessite : pip install -U trl transformers peft
    datasets accelerate (mêmes dépendances que dpo_train.py).
    """

    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: RewardModelRegistry | None = None, use_lora: bool = True):
        self.base_model_name = base_model
        self.registry = registry or RewardModelRegistry()
        self.use_lora = use_lora

    def run_cycle(self, store: FeedbackStore, index: EmbeddingIndex,
                  min_examples: int = 20, epochs: int = 1, learning_rate: float = 1e-5,
                  top_k: int = 3, fast_test: bool = False) -> RewardModelVersion:
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from trl import RewardConfig, RewardTrainer

        import os
        import torch
        has_cuda = torch.cuda.is_available()
        if not has_cuda:
            torch.set_num_threads(max(1, os.cpu_count() or 1))

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        examples = build_preference_dataset(
            store, index, tokenizer, min_examples=min_examples, top_k=top_k
        )
        cols = training_columns(examples)

        # RewardTrainer note le texte COMPLET (prompt + réponse), pas juste la
        # réponse seule — le score doit refléter "est-ce une bonne réponse à
        # CETTE question", pas juste "ce texte est-il bien écrit".
        for row in cols:
            row["chosen"] = row["prompt"] + row["chosen"]
            row["rejected"] = row["prompt"] + row["rejected"]

        train_dataset = Dataset.from_list([{"chosen": r["chosen"], "rejected": r["rejected"]} for r in cols])

        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, num_labels=1, torch_dtype=torch.float32
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        peft_config = None
        if self.use_lora:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS",
            )

        new_version = self.registry.next_version()
        output_dir = str(Path("reward_models") / f"v{new_version}")

        config = RewardConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=1 if fast_test else 2,
            gradient_accumulation_steps=1 if fast_test else 4,
            max_steps=3 if fast_test else -1,
            max_length=768,
            logging_steps=1,
            save_strategy="epoch",
            report_to=[],
            use_cpu=not has_cuda,
        )

        trainer = RewardTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        rv = RewardModelVersion(
            version=new_version,
            base_model=self.base_model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            n_training_pairs=len(examples),
            path=output_dir,
        )
        self.registry.register(rv)
        return rv
