"""
ppo_train.py — Cycle d'entraînement RL classique (PPO) pour comparaison avec
DPO et GRPO.

Différence clé avec dpo_train.py : PPO ne rejoue pas les paires (chosen,
rejected) telles quelles. Il GÉNÈRE des réponses à partir des prompts
pendant l'entraînement, les fait noter par le reward model (reward_model.py),
et met à jour la politique pour maximiser cette note tout en restant proche
du modèle de référence (pénalité KL). C'est plus coûteux que DPO (génération
+ notation + mise à jour à chaque step, au lieu d'un simple forward sur des
textes déjà écrits) — attendez-vous à un entraînement significativement plus
lent, même sur GPU.

Nécessite : pip install -U trl transformers peft datasets accelerate
Un reward model DOIT déjà exister (voir reward_model.py) avant de lancer un
cycle PPO — sans lui, PPO n'a aucun signal pour savoir quelles générations
sont bonnes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dataset_builder import build_preference_dataset, prompt_pool
from dpo_train import ModelRegistry, ModelVersion
from feedback_store import FeedbackStore
from minirag import EmbeddingIndex
from reward_model import RewardModelRegistry


class PPOTrainingPipeline:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: ModelRegistry | None = None,
                 reward_registry: RewardModelRegistry | None = None):
        self.base_model_name = base_model
        self.registry = registry or ModelRegistry()
        self.reward_registry = reward_registry or RewardModelRegistry()

    def run_cycle(self, store: FeedbackStore, index: EmbeddingIndex,
                  min_examples: int = 20, epochs: int = 1, learning_rate: float = 1e-6,
                  top_k: int = 3, fast_test: bool = False) -> ModelVersion:
        reward_entry = self.reward_registry.latest()
        if reward_entry is None:
            raise ValueError(
                "Aucun reward model entraîné — lancez d'abord un cycle "
                "reward_model.RewardModelPipeline.run_cycle() (voir POST /api/train/reward-model)."
            )

        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        from trl import PPOConfig, PPOTrainer

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
        prompts = prompt_pool(examples)

        # PPOTrainer attend un dataset tokenisé côté prompt.
        def tokenize_row(row):
            ids = tokenizer(row["prompt"], truncation=True, max_length=512)["input_ids"]
            return {"input_ids": ids}

        train_dataset = Dataset.from_list(prompts).map(tokenize_row, remove_columns=["prompt", "question"])

        policy = AutoModelForCausalLM.from_pretrained(self.base_model_name, torch_dtype=torch.float32)
        ref_policy = AutoModelForCausalLM.from_pretrained(self.base_model_name, torch_dtype=torch.float32)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_entry["path"], num_labels=1, torch_dtype=torch.float32
        )
        value_model = AutoModelForSequenceClassification.from_pretrained(
            reward_entry["path"], num_labels=1, torch_dtype=torch.float32
        )

        new_version = self.registry.next_version()
        output_dir = str(Path("models") / f"v{new_version}")

        config = PPOConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=1 if fast_test else 2,
            batch_size=4 if fast_test else 16,
            mini_batch_size=1 if fast_test else 4,
            response_length=128 if fast_test else 256,
            logging_steps=1,
            report_to=[],
            use_cpu=not has_cuda,
        )

        trainer = PPOTrainer(
            args=config,
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=train_dataset,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        mv = ModelVersion(
            version=new_version,
            base_model=self.base_model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_version=self.registry.next_version() - 1,  # pas de dataset figé côté PPO (génération live)
            n_training_pairs=len(examples),
            path=output_dir,
            status="staged",
            method="ppo",
        )
        self.registry.register(mv)
        return mv
