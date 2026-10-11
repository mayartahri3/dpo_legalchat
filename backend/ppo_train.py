"""
ppo_train.py — PPO (best-effort).

PPO est la méthode la plus lourde : politique + value model + reward model en
mémoire, et l'API de trl a changé plusieurs fois (PPOTrainer a été déplacé hors
du namespace principal dans les versions récentes). Les imports ci-dessous
essaient les deux emplacements ; si votre version de trl a encore changé,
l'erreur sera explicite plutôt que silencieuse. Je n'ai pas pu tester ce fichier
contre votre version installée de trl.
"""

from __future__ import annotations

from pathlib import Path

from dataset_builder import build_preference_dataset, prompt_pool
from feedback_store import FeedbackStore
from reward_model import RewardModelRegistry
from train_common import (
    ModelRegistry, ModelVersion, free_memory, load_merged, load_tokenizer, lora_config,
    make_config, register_trained, resolve_training_dtype,
)


def _import_ppo():
    try:
        from trl import PPOConfig, PPOTrainer
    except ImportError:
        try:
            from trl.experimental.ppo import PPOConfig, PPOTrainer
        except ImportError as e:
            raise ImportError(
                "PPOConfig/PPOTrainer introuvables dans votre version de trl "
                "(ni `trl` ni `trl.experimental.ppo`). Vérifiez `pip show trl`."
            ) from e
    return PPOConfig, PPOTrainer


class PPOTrainingPipeline:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: ModelRegistry | None = None,
                 reward_registry: RewardModelRegistry | None = None, use_lora: bool = True):
        self.base_model_name = base_model
        self.registry = registry or ModelRegistry()
        self.reward_registry = reward_registry or RewardModelRegistry()
        self.use_lora = use_lora

    def run_cycle(self, store: FeedbackStore, index, min_examples: int = 20, epochs: int = 1,
                  learning_rate: float = 1e-6, top_k: int = 3, fast_test: bool = False) -> ModelVersion:
        reward_entry = self.reward_registry.latest()
        if reward_entry is None:
            raise ValueError("Aucun reward model entraîné — lancez d'abord un cycle method='reward_model'.")

        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
        PPOConfig, PPOTrainer = _import_ppo()

        tokenizer = load_tokenizer(self.base_model_name)
        examples = build_preference_dataset(store, index, tokenizer, min_examples=min_examples, top_k=top_k)

        def tokenize(row):
            return {"input_ids": tokenizer(row["prompt"], truncation=True, max_length=512)["input_ids"]}

        train_dataset = Dataset.from_list(prompt_pool(examples)).map(
            tokenize, remove_columns=["prompt", "question"])

        dtype, has_cuda, _ = resolve_training_dtype()
        policy = AutoModelForCausalLM.from_pretrained(self.base_model_name, torch_dtype=dtype)
        ref_policy = None if self.use_lora else AutoModelForCausalLM.from_pretrained(
            self.base_model_name, torch_dtype=dtype)
        rm_args = (reward_entry["path"], reward_entry.get("base_model", self.base_model_name), dtype)
        reward_model = load_merged(AutoModelForSequenceClassification, *rm_args, num_labels=1)
        value_model = load_merged(AutoModelForSequenceClassification, *rm_args, num_labels=1)

        new_version = self.registry.next_version()
        output_dir = str(Path("models") / f"v{new_version}")
        config = make_config(
            PPOConfig, "ppo",
            output_dir=output_dir, num_train_epochs=epochs, learning_rate=learning_rate,
            per_device_train_batch_size=1 if fast_test else 2,
            batch_size=4 if fast_test else 16, mini_batch_size=1 if fast_test else 4,
            response_length=128 if fast_test else 256,
            logging_steps=1, save_strategy="no", report_to=[], use_cpu=not has_cuda,
        )
        kwargs = dict(args=config, processing_class=tokenizer, model=policy, ref_model=ref_policy,
                      reward_model=reward_model, value_model=value_model, train_dataset=train_dataset)
        if self.use_lora:
            kwargs["peft_config"] = lora_config("CAUSAL_LM")

        trainer = PPOTrainer(**kwargs)
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        mv = register_trained(self.registry, base_model=self.base_model_name, version=new_version,
                              path=output_dir, n_pairs=len(examples), method="ppo")
        del trainer, policy, ref_policy, reward_model, value_model
        free_memory()
        return mv