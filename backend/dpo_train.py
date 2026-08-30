"""
dpo_train.py — Reward signal + cycle d'entraînement DPO + versioning (§5.4, §7).

Périmètre (§3.1 / §3.2 / §6-Portabilité) :
  - Cible UNIQUEMENT un modèle open source contrôlé par E-Tafakna
    (ex. Qwen/Mistral/Llama — ici on réutilise le modèle de minirag.py),
    jamais un modèle propriétaire via API.
  - Le "Reward Model" du §5.4 est ici IMPLICITE : DPO n'entraîne pas de
    tête de récompense séparée, la log-vraisemblance relative
    policy/référence sur (chosen, rejected) joue ce rôle. C'est la
    différence fondamentale entre DPO et RLHF classique (PPO + Reward
    Model explicite) — voir le glossaire du cahier des charges, qui
    présente DPO comme "plus simple à mettre en œuvre que le RL classique".
    Un ReferenceRewardScorer optionnel est fourni plus bas pour l'étape
    d'évaluation comparative (§8 — dégradation entre versions).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dataset_builder import build_preference_dataset, save_dataset, training_columns
from feedback_store import FeedbackStore


# --------------------------------------------------------------------------- #
# Model registry — versioning + historique (§5.4 "Versionner chaque nouvelle
# version du modèle et conserver un historique des cycles d'entraînement")
# --------------------------------------------------------------------------- #

@dataclass
class ModelVersion:
    version: int
    base_model: str
    created_at: str
    dataset_version: int
    n_training_pairs: int
    path: str
    status: str = "staged"     # staged | active | retired
    method: str = "dpo"        # dpo | ppo | grpo — pour comparer les 3 approches (voir model_eval.py)
    eval_summary: dict = field(default_factory=dict)


class ModelRegistry:
    """Historique des versions du modèle open source entraîné par DPO."""

    def __init__(self, root: str | Path = "models"):
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

    def register(self, mv: ModelVersion) -> None:
        entries = self._load()
        entries.append(mv.__dict__)
        self._save(entries)

    def active_version(self) -> dict | None:
        entries = [e for e in self._load() if e["status"] == "active"]
        return entries[-1] if entries else None

    def promote(self, version: int) -> None:
        """Remplace la version en production après évaluation comparative (§7 étape 8, §8)."""
        entries = self._load()
        for e in entries:
            if e["status"] == "active":
                e["status"] = "retired"
            if e["version"] == version:
                e["status"] = "active"
        self._save(entries)

    def history(self) -> list[dict]:
        return self._load()

    def set_eval_summary(self, version: int, summary: dict) -> None:
        """Enregistre le résultat d'une comparaison (model_eval.py) sur une version
        donnée — permet au dashboard d'afficher les métriques sans tout recalculer."""
        entries = self._load()
        for e in entries:
            if e["version"] == version:
                e["eval_summary"] = summary
        self._save(entries)


# --------------------------------------------------------------------------- #
# DPO training pipeline
# --------------------------------------------------------------------------- #

class DPOTrainingPipeline:
    """
    Fine-tune le modèle open source contrôlé (LocalLLM de minirag.py) sur le
    dataset de préférences, via trl.DPOTrainer.

    Nécessite : pip install trl transformers datasets peft (optionnel: LoRA
    pour rester léger — recommandé, minirag.py cible des modèles 1-8B).
    """

    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: ModelRegistry | None = None, use_lora: bool = True):
        self.base_model_name = base_model
        self.registry = registry or ModelRegistry()
        self.use_lora = use_lora

    def _resolve_base_checkpoint(self) -> str:
        """Repart de la dernière version active si elle existe, sinon du modèle de base."""
        active = self.registry.active_version()
        return active["path"] if active else self.base_model_name

    def run_cycle(self, store: FeedbackStore, index: "EmbeddingIndex", dataset_dir: str | Path = "datasets",
                  min_examples: int = 20, epochs: int = 1, learning_rate: float = 5e-6,
                  top_k: int = 3, fast_test: bool = False) -> ModelVersion:
        """
        Un cycle complet §7 (étapes 5-7) :
          corrections validées -> dataset -> entraînement DPO -> nouvelle version.

        `index` est nécessaire pour reconstruire le contexte RAG des prompts
        d'entraînement (voir dataset_builder.build_preference_dataset) —
        sans lui, le modèle serait entraîné sur des prompts sans contexte,
        ce qui ne correspond pas à ce qu'il voit réellement en inférence.

        Lève ValueError si le volume de corrections validées est insuffisant
        (§8 — éviter le sur-apprentissage sur peu d'exemples).
        """
        from datasets import Dataset          # import tardif : dépendance lourde optionnelle
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer

        base_ckpt = self._resolve_base_checkpoint()
        tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        examples = build_preference_dataset(store, index, tokenizer, min_examples=min_examples, top_k=top_k)
        dataset_version = self.registry.next_version()
        dataset_path = save_dataset(examples, dataset_dir, dataset_version)

        model = AutoModelForCausalLM.from_pretrained(base_ckpt)
        # Le modèle de référence de DPO reste figé sur les poids de départ du
        # cycle — trl le clone en interne si `ref_model=None` et que le modèle
        # n'a pas d'adaptateurs PEFT actifs.
        ref_model = None

        peft_config = None
        if self.use_lora:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
            )

        train_dataset = Dataset.from_list(training_columns(examples))

        new_version = self.registry.next_version()
        output_dir = str(Path("models") / f"v{new_version}")

        # TrainingArguments essaie de deviner bf16/GPU automatiquement, ce qui
        # plante sur une machine CPU-only ("doesn't support bf16/gpu"). On
        # détecte explicitement le matériel disponible plutôt que de laisser
        # trl/transformers deviner.
        import os
        import torch
        has_cuda = torch.cuda.is_available()
        has_bf16 = has_cuda and torch.cuda.is_bf16_supported()
        if not has_cuda:
            # Utilise tous les coeurs dispo pour les forward/backward CPU —
            # sinon PyTorch peut se limiter à 1 thread selon l'environnement.
            torch.set_num_threads(max(1, os.cpu_count() or 1))

        # Le coût d'un pas DPO scale avec le carré de la longueur de séquence.
        # Le contexte juridique (articles de loi complets) dans `chosen`/
        # `rejected` peut faire plusieurs centaines/milliers de tokens — on
        # tronque pour garder des pas rapides, surtout utile sur CPU.
        max_length = 256 if fast_test else 768
        max_prompt_length = 128 if fast_test else 384

        config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            # fast_test réduit le batch à 1/pas d'accumulation pour avoir un
            # retour quasi immédiat sur la mécanique du pipeline (job créé,
            # version enregistrée, promotion possible) sans attendre un vrai
            # cycle d'entraînement complet.
            per_device_train_batch_size=1 if fast_test else 2,
            gradient_accumulation_steps=1 if fast_test else 4,
            max_steps=3 if fast_test else -1,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            beta=0.1,                       # force du terme de préférence DPO
            logging_steps=1,                # log chaque pas — avec peu d'exemples, logging_steps=10
                                             # ne loguait jamais rien avant la fin du run
            save_strategy="epoch",
            report_to=[],
            use_cpu=not has_cuda,
            bf16=has_bf16,
            fp16=False,                     # fp16 instable pour DPO sur petits modèles CPU/mixte
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        mv = ModelVersion(
            version=new_version,
            base_model=base_ckpt,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_version=dataset_version,
            n_training_pairs=len(examples),
            path=output_dir,
            status="staged",   # reste "staged" tant que l'évaluation comparative (§8) n'a pas eu lieu
            method="dpo",
        )
        self.registry.register(mv)
        return mv

    def should_trigger(self, store: FeedbackStore, min_examples: int = 20) -> bool:
        """
        Déclencheur simple pour un cycle "hebdomadaire/mensuel" (§7) : on
        planifie l'appel de cette méthode via un scheduler externe (cron,
        Airflow...) — ce module ne fait qu'exposer la condition métier.

        Ne reconstruit PAS les prompts (pas besoin d'index/tokenizer ici) —
        compte juste les paires (chosen, rejected) utilisables, comme le
        ferait build_preference_dataset avant l'étape de reconstruction RAG.
        """
        rows = store.validated_corrections()
        usable = sum(
            1 for row in rows
            if (row["final_answer"] or row["proposed_answer"])
            and row["rejected_answer"]
            and (row["final_answer"] or row["proposed_answer"]).strip() != row["rejected_answer"].strip()
        )
        return usable >= min_examples


# --------------------------------------------------------------------------- #
# Évaluation comparative avant remplacement — §7 étape 8, §8 (dégradation)
# --------------------------------------------------------------------------- #

class ReferenceRewardScorer:
    """
    Scoreur heuristique utilisé UNIQUEMENT pour comparer deux versions du
    modèle sur un jeu de questions de benchmark avant promotion (§8), pas
    pour l'entraînement lui-même (DPO n'a pas besoin de ce scoreur, cf.
    note en tête de fichier).

    Score = similarité d'embedding entre la réponse générée et la réponse
    "chosen" de référence pour la même question, quand elle existe dans le
    dataset de préférences validé. Une vraie évaluation juridique reste
    humaine (§3.2 — exclue du périmètre technique).
    """

    def __init__(self, embed_model):
        self.embed_model = embed_model

    def score(self, generated: str, reference_chosen: str) -> float:
        import numpy as np
        vecs = self.embed_model.encode([generated, reference_chosen], normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))

    def compare_versions(self, questions_and_refs: list[tuple[str, str]],
                          generate_fn_a, generate_fn_b) -> dict:
        """
        generate_fn_a / generate_fn_b : callables(question) -> str, typiquement
        `LocalLLM.generate` chargé sur chacune des deux versions du modèle.
        Renvoie un résumé exploitable pour la décision de promotion (§7-8).
        """
        scores_a, scores_b = [], []
        for question, reference in questions_and_refs:
            scores_a.append(self.score(generate_fn_a(question), reference))
            scores_b.append(self.score(generate_fn_b(question), reference))

        mean_a = sum(scores_a) / len(scores_a) if scores_a else 0.0
        mean_b = sum(scores_b) / len(scores_b) if scores_b else 0.0
        return {
            "n_questions": len(questions_and_refs),
            "mean_score_a": round(mean_a, 4),
            "mean_score_b": round(mean_b, 4),
            "b_wins": mean_b > mean_a,
            "delta": round(mean_b - mean_a, 4),
        }
