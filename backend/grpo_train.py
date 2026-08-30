"""
grpo_train.py — Group Relative Policy Optimization (méthode derrière
DeepSeek-R1), troisième approche pour la comparaison DPO / PPO / GRPO.

Avantage pratique par rapport à PPO : PAS BESOIN d'un reward model appris ni
d'un value model séparé. GRPO génère plusieurs complétions par prompt (un
"groupe"), les note avec une fonction de récompense (qui peut être une
simple heuristique Python), et pousse la politique vers les complétions du
groupe qui obtiennent un score au-dessus de la moyenne du groupe. C'est
notablement plus léger à mettre en place que PPO, au prix de générer
plusieurs complétions par prompt à chaque step (donc plus de calcul par
step que DPO).

Fonction de récompense utilisée ici : similarité d'embedding (cosine) entre
la complétion générée et la réponse validée ("chosen") pour cette question
— même principe que ReferenceRewardScorer dans dpo_train.py, réutilisé
directement comme signal d'entraînement plutôt que juste comme outil
d'évaluation a posteriori.

Nécessite : pip install -U trl transformers peft datasets accelerate
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dataset_builder import build_preference_dataset
from dpo_train import ModelRegistry, ModelVersion
from feedback_store import FeedbackStore
from minirag import EmbeddingIndex


class GRPOTrainingPipeline:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: ModelRegistry | None = None):
        self.base_model_name = base_model
        self.registry = registry or ModelRegistry()

    def run_cycle(self, store: FeedbackStore, index: EmbeddingIndex,
                  min_examples: int = 20, epochs: int = 1, learning_rate: float = 1e-6,
                  top_k: int = 3, num_generations: int = 4, fast_test: bool = False) -> ModelVersion:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

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
        # question -> réponse validée, pour que la fonction de récompense sache
        # à quoi comparer chaque complétion générée pendant l'entraînement.
        reference_by_prompt = {e["prompt"]: e["chosen"] for e in examples}

        train_dataset = Dataset.from_list([{"prompt": e["prompt"]} for e in examples])

        embed_model = index.model  # même modèle sentence-transformers que la recherche RAG

        def similarity_reward(completions, prompts, **kwargs):
            """Fonction de récompense GRPO — cosine similarity(complétion, réponse validée).
            Signature attendue par trl.GRPOTrainer : reçoit les listes de complétions
            générées et les prompts correspondants, renvoie une liste de floats."""
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

        config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=1 if fast_test else 2,
            num_generations=2 if fast_test else num_generations,  # taille du "groupe" comparé par prompt
            max_completion_length=128 if fast_test else 300,
            max_steps=3 if fast_test else -1,
            logging_steps=1,
            report_to=[],
            use_cpu=not has_cuda,
        )

        trainer = GRPOTrainer(
            model=self.base_model_name,
            reward_funcs=similarity_reward,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        trainer.train()
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
        return mv
