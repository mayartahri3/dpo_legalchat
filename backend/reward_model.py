"""
reward_model.py — Reward model explicite (nécessaire à PPO, optionnel pour GRPO).
Tête de classification (1 score) sur le même modèle de base, entraînée sur les
paires (chosen, rejected) via trl.RewardTrainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dataset_builder import build_preference_dataset, training_columns
from feedback_store import FeedbackStore
from train_common import (
    JsonRegistry, free_memory, load_tokenizer, lora_config, make_config, resolve_training_dtype,
)


@dataclass
class RewardModelVersion:
    version: int
    base_model: str
    created_at: str
    n_training_pairs: int
    path: str
    status: str = "ready"


class RewardModelRegistry(JsonRegistry):
    def __init__(self, root: str | Path = "reward_models"):
        super().__init__(root)

    def latest(self) -> dict | None:
        entries = self._load()
        return entries[-1] if entries else None


class RewardModelPipeline:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: RewardModelRegistry | None = None, use_lora: bool = True):
        self.base_model_name = base_model
        self.registry = registry or RewardModelRegistry()
        self.use_lora = use_lora

    def run_cycle(self, store: FeedbackStore, index, min_examples: int = 20, epochs: int = 1,
                  learning_rate: float = 1e-5, top_k: int = 3,
                  fast_test: bool = False) -> RewardModelVersion:
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification
        from trl import RewardConfig, RewardTrainer

        tokenizer = load_tokenizer(self.base_model_name)
        examples = build_preference_dataset(store, index, tokenizer, min_examples=min_examples, top_k=top_k)

        # Le score doit juger "bonne réponse à CETTE question" => on note prompt + réponse.
        rows = [{"chosen": r["prompt"] + r["chosen"], "rejected": r["prompt"] + r["rejected"]}
                for r in training_columns(examples)]

        dtype, has_cuda, has_bf16 = resolve_training_dtype()
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, num_labels=1, torch_dtype=dtype)
        model.config.pad_token_id = tokenizer.pad_token_id

        new_version = self.registry.next_version()
        output_dir = str(Path("reward_models") / f"v{new_version}")
        config = make_config(
            RewardConfig, "reward",
            output_dir=output_dir, num_train_epochs=epochs, learning_rate=learning_rate,
            per_device_train_batch_size=1 if fast_test else 2,
            gradient_accumulation_steps=1 if fast_test else 4,
            max_steps=3 if fast_test else -1, max_length=768,
            logging_steps=1, save_strategy="no", report_to=[], use_cpu=not has_cuda,
            bf16=has_bf16, fp16=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        trainer = RewardTrainer(
            model=model, args=config, train_dataset=Dataset.from_list(rows),
            processing_class=tokenizer,
            peft_config=lora_config("SEQ_CLS") if self.use_lora else None,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        rv = RewardModelVersion(
            version=new_version, base_model=self.base_model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            n_training_pairs=len(examples), path=output_dir,
        )
        self.registry.register(rv)
        del trainer, model
        free_memory()
        return rv