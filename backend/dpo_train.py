"""
dpo_train.py — Reward signal + cycle d'entraînement DPO + versioning (§5.4, §7).

VERSION OPTIMISÉE RAM/VRAM COLAB (voir CHANGES en bas de fichier pour le
détail de ce qui a changé par rapport à la version d'origine).

Périmètre (§3.1 / §3.2 / §6-Portabilité) :
  - Cible UNIQUEMENT un modèle open source contrôlé par E-Tafakna
    (ex. Qwen/Mistral/Llama — ici on réutilise le modèle de minirag.py),
    jamais un modèle propriétaire via API.
  - Le "Reward Model" du §5.4 est ici IMPLICITE : DPO n'entraîne pas de
    tête de récompense séparée, la log-vraisemblance relative
    policy/référence sur (chosen, rejected) joue ce rôle.
"""

from __future__ import annotations

import gc
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dataset_builder import build_preference_dataset, save_dataset, training_columns
from feedback_store import FeedbackStore


# --------------------------------------------------------------------------- #
# Model registry — versioning + historique (§5.4)
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
    method: str = "dpo"        # dpo | ppo | grpo
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
        entries = self._load()
        for e in entries:
            if e["version"] == version:
                e["eval_summary"] = summary
        self._save(entries)


# --------------------------------------------------------------------------- #
# Memory helpers — utilisés par dpo_train.py ET grpo_train.py
# --------------------------------------------------------------------------- #

def resolve_training_dtype():
    """Choisit le dtype le plus économe en mémoire disponible sur ce matériel.

    fp32 pour les poids d'un modèle 1.5B ~= 6 Go rien que pour les poids.
    bf16/fp16 divise ça par 2 — la différence entre "ça tient sur un T4 16 Go"
    et "ça OOM avant même la première step". Sur CPU (pas de GPU Colab
    attribué) on reste en fp32, torch ne supporte pas bien le matmul bf16/fp16
    sur CPU pour l'entraînement.
    """
    import torch
    if not torch.cuda.is_available():
        return torch.float32, False, False
    has_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if has_bf16 else torch.float16
    return dtype, True, has_bf16


def print_gpu_mem(tag: str) -> None:
    import torch
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[mem] {tag}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


def free_memory(*objs) -> None:
    """Supprime les références passées puis force le garbage collector +
    la libération du cache CUDA. À appeler après chaque run_cycle avant de
    lancer un autre cycle dans le MÊME process Colab (sinon les allocateurs
    CUDA gardent la mémoire réservée même après suppression des objets)."""
    import torch
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# DPO training pipeline
# --------------------------------------------------------------------------- #

class DPOTrainingPipeline:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 registry: ModelRegistry | None = None, use_lora: bool = True):
        self.base_model_name = base_model
        self.registry = registry or ModelRegistry()
        self.use_lora = use_lora

    def _resolve_base_checkpoint(self) -> str:
        active = self.registry.active_version()
        return active["path"] if active else self.base_model_name

    def run_cycle(self, store: FeedbackStore, index: "EmbeddingIndex", dataset_dir: str | Path = "datasets",
                  min_examples: int = 20, epochs: int = 1, learning_rate: float = 5e-6,
                  top_k: int = 3, fast_test: bool = False,
                  per_device_train_batch_size: int = 1,
                  gradient_accumulation_steps: int = 8) -> ModelVersion:
        """
        Un cycle complet §7 (étapes 5-7).

        CHANGÉ vs original : batch_size=1 / grad_accum=8 par défaut (au lieu
        de batch_size=2 / accum=4) — même throughput effectif (8 séquences
        par update), mais le pic mémoire d'activation ne dépend que du
        batch PHYSIQUE (1), pas du batch effectif. C'est le levier le plus
        direct contre un OOM Colab sans changer la qualité d'entraînement.
        """
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        import dataclasses
        import os
        import torch

        base_ckpt = self._resolve_base_checkpoint()
        tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        examples = build_preference_dataset(store, index, tokenizer, min_examples=min_examples, top_k=top_k)
        dataset_version = self.registry.next_version()
        dataset_path = save_dataset(examples, dataset_dir, dataset_version)

        has_cuda = torch.cuda.is_available()
        model_dtype, _, has_bf16 = resolve_training_dtype()
        if not has_cuda:
            torch.set_num_threads(max(1, os.cpu_count() or 1))

        print_gpu_mem("before model load")
        model = AutoModelForCausalLM.from_pretrained(base_ckpt, torch_dtype=model_dtype)
        # Requis dès qu'on active gradient checkpointing plus bas — sinon le
        # modèle garde le cache d'attention en mémoire pour rien pendant le
        # forward d'entraînement (use_cache ne sert qu'à la génération).
        model.config.use_cache = False
        print_gpu_mem("after model load")

        ref_model = None  # trl clone en interne (adaptateurs désactivés) — pas de 2e copie du modèle

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

        max_length = 256 if fast_test else 768
        max_prompt_length = 128 if fast_test else 384

        # CHANGÉ — robustesse aux versions de trl : `pip install -U trl` peut
        # attraper une release plus récente que celle testée ici, et trl a
        # déjà supprimé/renommé des champs de *Config d'une version à l'autre
        # (ex: max_prompt_length déprécié puis retiré de DPOConfig). Plutôt
        # que de planter sur un TypeError, on ne passe QUE les kwargs que la
        # version installée accepte réellement, et on log ce qui est ignoré.
        requested_config_kwargs = dict(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=1 if fast_test else per_device_train_batch_size,
            gradient_accumulation_steps=1 if fast_test else gradient_accumulation_steps,
            max_steps=3 if fast_test else -1,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            beta=0.1,
            logging_steps=1,
            save_strategy="epoch",
            report_to=[],
            use_cpu=not has_cuda,
            bf16=has_bf16,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
        )
        accepted_fields = {f.name for f in dataclasses.fields(DPOConfig)}
        dropped = sorted(k for k in requested_config_kwargs if k not in accepted_fields)
        if dropped:
            print(f"[dpo_train] ⚠ Cette version de trl n'accepte pas ces options DPOConfig, "
                  f"elles sont ignorées : {dropped}")
            if "max_prompt_length" in dropped:
                print("  (max_prompt_length déprécié dans les trl récents — seul max_length "
                      "s'applique désormais, avec troncature 'keep_start' par défaut.)")
        config_kwargs = {k: v for k, v in requested_config_kwargs.items() if k in accepted_fields}
        config = DPOConfig(**config_kwargs)

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        print_gpu_mem("before train")
        trainer.train()
        print_gpu_mem("after train")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        mv = ModelVersion(
            version=new_version,
            base_model=base_ckpt,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_version=dataset_version,
            n_training_pairs=len(examples),
            path=output_dir,
            status="staged",
            method="dpo",
        )
        self.registry.register(mv)

        # CHANGÉ : libère explicitement le modèle/trainer avant de rendre la
        # main — sans ça, si colab_train.py enchaîne un 2e cycle dans le
        # même process, la mémoire du premier cycle reste réservée par
        # l'allocateur CUDA et peut faire échouer le 2e cycle même s'il est
        # individuellement plus petit.
        free_memory(trainer, model, ref_model)
        print_gpu_mem("after cleanup")

        return mv

    def should_trigger(self, store: FeedbackStore, min_examples: int = 20) -> bool:
        rows = store.validated_corrections()
        usable = sum(
            1 for row in rows
            if (row["final_answer"] or row["proposed_answer"])
            and row["rejected_answer"]
            and (row["final_answer"] or row["proposed_answer"]).strip() != row["rejected_answer"].strip()
        )
        return usable >= min_examples


# --------------------------------------------------------------------------- #
# Évaluation comparative avant remplacement — §7 étape 8, §8
# --------------------------------------------------------------------------- #

class ReferenceRewardScorer:
    def __init__(self, embed_model):
        self.embed_model = embed_model

    def score(self, generated: str, reference_chosen: str) -> float:
        import numpy as np
        vecs = self.embed_model.encode([generated, reference_chosen], normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))

    def compare_versions(self, questions_and_refs: list[tuple[str, str]],
                          generate_fn_a, generate_fn_b) -> dict:
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


# --------------------------------------------------------------------------- #
# CHANGES vs. version d'origine
# --------------------------------------------------------------------------- #
# 1. Modèle chargé en bf16 (ou fp16 si bf16 indisponible) au lieu de fp32
#    systématique -> ~2x moins de mémoire pour les poids du modèle de base.
# 2. gradient_checkpointing=True + use_cache=False -> réduit fortement la
#    mémoire d'activations au prix d'un peu de temps de calcul en plus
#    (recompute des activations au backward).
# 3. per_device_train_batch_size=1 par défaut avec gradient_accumulation
#    plus élevé -> même throughput effectif, pic mémoire plus bas.
# 4. free_memory() explicite en fin de run_cycle -> permet d'enchaîner
#    plusieurs cycles dans le même process Colab sans fuite d'allocation
#    CUDA (utile si vous scriptez DPO puis GRPO dans le même notebook —
#    même si le guide recommande un restart runtime entre les deux par
#    prudence supplémentaire).
# 5. print_gpu_mem() à des points clés -> permet de voir exactement où la
#    mémoire grimpe si un OOM survient malgré tout, au lieu de deviner.
# 6. DPOConfig(**kwargs) filtré dynamiquement sur dataclasses.fields() ->
#    évite un crash sec (TypeError) si `pip install -U trl` attrape une
#    version qui a retiré/renommé un champ (ex: max_prompt_length) entre le
#    moment où ce script a été écrit et le moment où vous le lancez.