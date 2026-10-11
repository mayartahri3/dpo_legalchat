"""
grpo_train.py — GRPO (LoRA + bf16/fp16 + gradient checkpointing). Pas de reward
model : la récompense est la similarité d'embedding entre la complétion et la
réponse validée.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_builder import build_preference_dataset
from feedback_store import FeedbackStore
from train_common import (
    ModelRegistry, ModelVersion, free_memory, load_tokenizer, lora_config, make_config,
    print_gpu_mem, register_trained, resolve_training_dtype,
)


class GRPOTrainingPipeline:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: ModelRegistry | None = None, use_lora: bool = True):
        self.base_model_name = base_model
        self.registry = registry or ModelRegistry()
        self.use_lora = use_lora

    def run_cycle(self, store: FeedbackStore, index, min_examples: int = 20, epochs: int = 1,
                  learning_rate: float = 1e-5, top_k: int = 3, num_generations: int = 2,
                  fast_test: bool = False, per_device_train_batch_size: int = 1,
                  max_completion_length: int = 200,
                  max_prompt_length: int = 1024) -> ModelVersion:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        tokenizer = load_tokenizer(self.base_model_name)
        examples = build_preference_dataset(store, index, tokenizer, min_examples=min_examples, top_k=top_k)
        reference_by_prompt = {e["prompt"]: e["chosen"] for e in examples}
        embed = index.model

        def similarity_reward(completions, prompts, **_):
            refs = [reference_by_prompt.get(p, "") for p in prompts]
            vecs = embed.encode(list(completions) + refs, normalize_embeddings=True)
            n = len(completions)
            sims = (vecs[:n] * vecs[n:]).sum(axis=1)
            return [float(s) if ref else 0.0 for s, ref in zip(sims, refs)]

        dtype, has_cuda, has_bf16 = resolve_training_dtype()
        new_version = self.registry.next_version()
        output_dir = str(Path("models") / f"v{new_version}")

        batch = 1 if fast_test else per_device_train_batch_size
        n_gen = 2 if fast_test else num_generations
        config = make_config(
            GRPOConfig, "grpo",
            output_dir=output_dir, num_train_epochs=epochs, learning_rate=learning_rate,
            per_device_train_batch_size=batch, num_generations=n_gen,
            generation_batch_size=batch * n_gen,   # doit être un multiple de num_generations
            max_completion_length=128 if fast_test else max_completion_length,
            max_prompt_length=max_prompt_length,
            max_steps=3 if fast_test else -1,
            logging_steps=1, save_strategy="no", report_to=[], use_cpu=not has_cuda, bf16=has_bf16,
            gradient_checkpointing=True, model_init_kwargs={"torch_dtype": dtype},
        )
        print_gpu_mem("before GRPO init")
        trainer = GRPOTrainer(
            model=self.base_model_name, reward_funcs=similarity_reward, args=config,
            train_dataset=Dataset.from_list([{"prompt": e["prompt"]} for e in examples]),
            processing_class=tokenizer,
            peft_config=lora_config("CAUSAL_LM") if self.use_lora else None,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        mv = register_trained(self.registry, base_model=self.base_model_name, version=new_version,
                              path=output_dir, n_pairs=len(examples), method="grpo")
        del trainer
        free_memory()
        return mv