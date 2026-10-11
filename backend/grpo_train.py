"""
grpo_train.py — Group Relative Policy Optimization (méthode derrière
DeepSeek-R1), troisième approche pour la comparaison DPO / PPO / GRPO.

VERSION CORRIGÉE — voir CHANGES en bas de fichier. Le point important :
la version d'origine de ce fichier n'appliquait PAS de LoRA (peft_config
n'était jamais passé à GRPOTrainer), contrairement à dpo_train.py et
reward_model.py. Ça revient à fine-tuner les 1.5 milliards de paramètres
du modèle en entier, avec les états d'optimizer Adam correspondants
(2 copies fp32 supplémentaires par paramètre entraînable) — c'est très
probablement la cause principale des crashs RAM/VRAM sur Colab, bien plus
que la taille de num_generations.

Avantage pratique par rapport à PPO : PAS BESOIN d'un reward model appris ni
d'un value model séparé. GRPO génère plusieurs complétions par prompt (un
"groupe"), les note avec une fonction de récompense (qui peut être une
simple heuristique Python), et pousse la politique vers les complétions du
groupe qui obtiennent un score au-dessus de la moyenne du groupe.

Fonction de récompense utilisée ici : similarité d'embedding (cosine) entre
la complétion générée et la réponse validée ("chosen") pour cette question.

Nécessite : pip install -U trl transformers peft datasets accelerate
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dataset_builder import build_preference_dataset
from dpo_train import ModelRegistry, ModelVersion, resolve_training_dtype, print_gpu_mem, free_memory
from feedback_store import FeedbackStore
from minirag import EmbeddingIndex


class GRPOTrainingPipeline:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: ModelRegistry | None = None, use_lora: bool = True):
        self.base_model_name = base_model
        self.registry = registry or ModelRegistry()
        self.use_lora = use_lora  # CHANGÉ : était absent — GRPO entraînait le modèle complet

    def run_cycle(self, store: FeedbackStore, index: EmbeddingIndex,
                  min_examples: int = 20, epochs: int = 1, learning_rate: float = 1e-6,
                  top_k: int = 3, num_generations: int = 2, fast_test: bool = False,
                  per_device_train_batch_size: int = 1,
                  max_completion_length: int = 200) -> ModelVersion:
        """
        CHANGÉ vs original :
          - num_generations par défaut réduit à 2 (était 4). Le coût mémoire
            de génération pendant l'entraînement scale avec
            batch * num_generations (chaque complétion garde son propre
            KV-cache pendant le rollout) — c'est le 2e plus gros levier
            mémoire après le LoRA manquant.
          - max_completion_length par défaut réduit à 200 (était 300).
          - peft_config (LoRA) maintenant appliqué, comme dpo_train.py.
          - model_init_kwargs force le dtype bf16/fp16 au lieu du fp32 par
            défaut de trl quand `model` est passé en string.
        """
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        import dataclasses
        import os
        import torch
        has_cuda = torch.cuda.is_available()
        model_dtype, _, has_bf16 = resolve_training_dtype()
        if not has_cuda:
            torch.set_num_threads(max(1, os.cpu_count() or 1))

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        examples = build_preference_dataset(
            store, index, tokenizer, min_examples=min_examples, top_k=top_k
        )
        reference_by_prompt = {e["prompt"]: e["chosen"] for e in examples}

        train_dataset = Dataset.from_list([{"prompt": e["prompt"]} for e in examples])

        embed_model = index.model

        def similarity_reward(completions, prompts, **kwargs):
            import numpy as np
            rewards = []
            for prompt, completion in zip(prompts, completions):
                reference = reference_by_prompt.get(prompt, "")
                if not reference:
                    rewards.append(0.0)
                    continue
                vecs = embed_model.encode([completion, reference], normalize_embeddings=True)
                rewards.append(float(np.dot(vecs[0], vecs[1])))
            return rewards

        new_version = self.registry.next_version()
        output_dir = str(Path("models") / f"v{new_version}")

        peft_config = None
        if self.use_lora:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
            )

        print_gpu_mem("before GRPO trainer init")

        # Valeurs effectives une fois fast_test appliqué — calculées ici pour
        # pouvoir dériver generation_batch_size correctement juste après
        # (voir note ci-dessous).
        effective_batch_size = 1 if fast_test else per_device_train_batch_size
        effective_num_generations = 2 if fast_test else num_generations

        # CHANGÉ — même robustesse que dpo_train.py : ne passe que les
        # kwargs que la version de trl installée accepte réellement pour
        # GRPOConfig, log le reste au lieu de planter sur un TypeError.
        requested_config_kwargs = dict(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=effective_batch_size,
            num_generations=effective_num_generations,
            # generation_batch_size DOIT être un multiple de num_generations
            # (contrainte imposée par trl.GRPOConfig.__post_init__) — le
            # défaut de trl le dérive de per_device_train_batch_size, ce qui
            # casse dès que batch_size < num_generations (notre cas : 1 < 2).
            # On le fixe explicitement ici pour être toujours valide, quel
            # que soit le batch_size choisi.
            generation_batch_size=effective_batch_size * effective_num_generations,
            max_completion_length=128 if fast_test else max_completion_length,
            max_steps=3 if fast_test else -1,
            logging_steps=1,
            report_to=[],
            use_cpu=not has_cuda,
            bf16=has_bf16,
            gradient_checkpointing=True,
            # GRPOTrainer charge le modèle en interne à partir du nom/chemin
            # (self.base_model_name est une string) — model_init_kwargs est
            # forwardé à AutoModelForCausalLM.from_pretrained(), sinon trl
            # charge en fp32 par défaut peu importe ce qu'on met ailleurs.
            model_init_kwargs={"torch_dtype": model_dtype},
        )
        accepted_fields = {f.name for f in dataclasses.fields(GRPOConfig)}
        dropped = sorted(k for k in requested_config_kwargs if k not in accepted_fields)
        if dropped:
            print(f"[grpo_train] ⚠ Cette version de trl n'accepte pas ces options GRPOConfig, "
                  f"elles sont ignorées : {dropped}")
        config_kwargs = {k: v for k, v in requested_config_kwargs.items() if k in accepted_fields}
        config = GRPOConfig(**config_kwargs)

        trainer = GRPOTrainer(
            model=self.base_model_name,
            reward_funcs=similarity_reward,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,   # CHANGÉ : était absent
        )
        print_gpu_mem("before train")
        trainer.train()
        print_gpu_mem("after train")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        mv = ModelVersion(
            version=new_version,
            base_model=self.base_model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_version=0,  # pas de dataset figé — récompense calculée à la volée
            n_training_pairs=len(examples),
            path=output_dir,
            status="staged",
            method="grpo",
        )
        self.registry.register(mv)

        free_memory(trainer)
        print_gpu_mem("after cleanup")

        return mv


# --------------------------------------------------------------------------- #
# CHANGES vs. version d'origine
# --------------------------------------------------------------------------- #
# 1. peft_config (LoRA, r=16) maintenant passé à GRPOTrainer — c'était
#    absent dans l'original, ce qui entraînait TOUS les paramètres du
#    modèle (1.5B) au lieu des ~quelques millions de paramètres LoRA.
#    C'est le changement le plus important de ce fichier.
# 2. model_init_kwargs={"torch_dtype": ...} force bf16/fp16 pour le
#    chargement interne du modèle par GRPOTrainer (sinon fp32 par défaut).
# 3. num_generations par défaut 2 au lieu de 4, max_completion_length 200
#    au lieu de 300 — réduit le pic mémoire du rollout de génération.
# 4. gradient_checkpointing=True ajouté (absent avant).
# 5. free_memory()/print_gpu_mem() — mêmes raisons que dpo_train.py.