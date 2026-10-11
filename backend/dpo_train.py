"""
dpo_train.py — Cycle DPO (LoRA) — §5.4, §7. DPO n'a pas besoin de reward model
séparé : la log-vraisemblance relative policy/référence joue ce rôle.

Les registries et helpers sont dans train_common.py (ré-exportés ici pour que
colab_train.py / colab_eval.py continuent de fonctionner).
"""

from __future__ import annotations

from pathlib import Path

from dataset_builder import build_preference_dataset, save_dataset, training_columns
from feedback_store import FeedbackStore
from train_common import (  # noqa: F401  (ré-exports)
    ModelRegistry, ModelVersion, count_usable_pairs, free_memory, load_tokenizer, lora_config,
    make_config, print_gpu_mem, register_trained, resolve_training_dtype,
)


class DPOTrainingPipeline:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: ModelRegistry | None = None, use_lora: bool = True):
        self.base_model_name = base_model
        self.registry = registry or ModelRegistry()
        self.use_lora = use_lora

    def run_cycle(self, store: FeedbackStore, index, dataset_dir: str | Path = "datasets",
                  min_examples: int = 20, epochs: int = 1, learning_rate: float = 5e-5,
                  top_k: int = 3, fast_test: bool = False,
                  per_device_train_batch_size: int = 1,
                  gradient_accumulation_steps: int = 4, max_length: int = 1536) -> ModelVersion:
        """Toujours entraîné depuis le modèle de BASE sur l'ensemble des corrections
        validées (pas d'empilement d'adaptateurs sur la version active).

        learning_rate 5e-5 : 5e-6 est adapté au full fine-tuning, pas à LoRA — avec
        ~20 paires cela ne changeait quasiment rien au modèle."""
        from datasets import Dataset
        from transformers import AutoModelForCausalLM
        from trl import DPOConfig, DPOTrainer

        tokenizer = load_tokenizer(self.base_model_name)
        examples = build_preference_dataset(store, index, tokenizer, min_examples=min_examples, top_k=top_k)
        dataset_version = self.registry.next_version()
        save_dataset(examples, dataset_dir, dataset_version)

        dtype, has_cuda, has_bf16 = resolve_training_dtype()
        print_gpu_mem("before model load")
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name, torch_dtype=dtype)
        model.config.use_cache = False   # inutile (et coûteux) avec gradient checkpointing

        output_dir = str(Path("models") / f"v{dataset_version}")
        config = make_config(
            DPOConfig, "dpo",
            output_dir=output_dir, num_train_epochs=epochs, learning_rate=learning_rate,
            per_device_train_batch_size=1 if fast_test else per_device_train_batch_size,
            gradient_accumulation_steps=1 if fast_test else gradient_accumulation_steps,
            max_steps=3 if fast_test else -1,
            max_length=max_length,
            max_prompt_length=max_length // 2,  # ignoré par les trl récents
            beta=0.1, logging_steps=1, save_strategy="no", report_to=[],
            use_cpu=not has_cuda, bf16=has_bf16, fp16=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
        )
        trainer = DPOTrainer(
            model=model, args=config, train_dataset=Dataset.from_list(training_columns(examples)),
            processing_class=tokenizer,
            peft_config=lora_config("CAUSAL_LM") if self.use_lora else None,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        mv = register_trained(self.registry, base_model=self.base_model_name, version=dataset_version,
                              path=output_dir, n_pairs=len(examples), method="dpo",
                              dataset_version=dataset_version)
        del trainer, model
        free_memory()
        return mv

    def should_trigger(self, store: FeedbackStore, min_examples: int = 20) -> bool:
        return count_usable_pairs(store) >= min_examples


class ReferenceRewardScorer:
    """Similarité cosinus générée/validée (conservé pour compatibilité ; voir eval_metrics.py)."""

    def __init__(self, embed_model):
        self.embed_model = embed_model

    def score(self, generated: str, reference_chosen: str) -> float:
        import numpy as np
        v = self.embed_model.encode([generated, reference_chosen], normalize_embeddings=True)
        return float(np.dot(v[0], v[1]))