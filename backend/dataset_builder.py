"""
dataset_builder.py — Constitution du dataset de préférences (§5.4).

Transforme les corrections validées (statut VALIDATED / AUTO_VALIDATED /
REFORMULATED) en paires (prompt, chosen, rejected) au format attendu par
trl.DPOTrainer, et journalise chaque version du dataset (traçabilité §6).

IMPORTANT : le prompt d'entraînement doit être identique en forme à ce que
LocalLLM.generate() construit réellement à l'inférence — contexte RAG
(articles récupérés) + apply_chat_template — sinon le DPO/PPO/GRPO entraîne
le modèle sur une distribution de prompts qu'il ne verra jamais en usage
réel. On ré-exécute donc EmbeddingIndex.search() pour chaque question au
moment de construire le dataset, exactement comme au moment où la réponse
originale a été générée.
"""

from __future__ import annotations

import json
from pathlib import Path

from feedback_store import FeedbackStore
from minirag import EmbeddingIndex, build_chat_prompt


SYSTEM_PROMPT = (
    "Tu es un assistant juridique. Réponds UNIQUEMENT à partir des articles "
    "de loi fournis en contexte. Si le contexte ne contient pas la réponse, "
    "dis-le clairement. Cite systématiquement le numéro d'article utilisé."
)


def build_preference_dataset(store: FeedbackStore, index: EmbeddingIndex, tokenizer,
                              min_examples: int = 1, top_k: int = 3,
                              system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """
    Renvoie une liste de dicts {"prompt", "chosen", "rejected"} — le format
    natif de trl.DPOTrainer (colonnes `prompt`, `chosen`, `rejected`), et
    réutilisable tel quel pour reward_model.py (RewardTrainer) et pour
    construire le pool de prompts de ppo_train.py / grpo_train.py.

    `index` et `tokenizer` reconstruisent le MÊME prompt (contexte RAG
    inclus) que celui vu par le modèle en inférence réelle — voir
    minirag.build_chat_prompt. `top_k` doit correspondre au TOP_K utilisé en
    production (api.py), sinon le contexte reconstruit ne correspond pas
    exactement à celui montré à l'origine.
    """
    rows = store.validated_corrections()
    examples = []
    for row in rows:
        chosen = row["final_answer"] or row["proposed_answer"]
        rejected = row["rejected_answer"]
        if not chosen or not rejected or chosen.strip() == rejected.strip():
            continue  # paire dégénérée — on ne renforce pas le "ne rien changer"

        hits = index.search(row["question"], top_k=top_k)
        context_articles = [a for a, _score in hits]
        prompt = build_chat_prompt(tokenizer, row["question"], context_articles, system_prompt)

        examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "question": row["question"],          # gardé pour ppo_train/grpo_train (pool de prompts) et model_eval
            "correction_id": row["id"],           # traçabilité, retiré avant l'entraînement
            "confidence_score": row["confidence_score"],
        })

    if len(examples) < min_examples:
        raise ValueError(
            f"Seulement {len(examples)} paire(s) validée(s) disponibles "
            f"(minimum requis: {min_examples}). Voir §8 — risque de "
            f"sur-apprentissage sur un faible nombre de corrections."
        )
    return examples


def save_dataset(examples: list[dict], out_dir: str | Path, version: int) -> Path:
    """Écrit dataset_v{version}.jsonl et conserve les métadonnées (source data, §6)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dataset_v{version}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    meta_path = out_dir / f"dataset_v{version}.meta.json"
    meta_path.write_text(json.dumps({
        "version": version,
        "n_examples": len(examples),
        "correction_ids": [ex["correction_id"] for ex in examples],
    }, indent=2), encoding="utf-8")
    return path


def training_columns(examples: list[dict]) -> list[dict]:
    """Vue épurée sans métadonnées internes — ce qu'on passe réellement à DPOTrainer/RewardTrainer."""
    return [{"prompt": e["prompt"], "chosen": e["chosen"], "rejected": e["rejected"]} for e in examples]


def prompt_pool(examples: list[dict]) -> list[dict]:
    """Vue {"prompt", "question"} — ce que consomment ppo_train.py / grpo_train.py, qui
    génèrent eux-mêmes des complétions à partir du prompt plutôt que de rejouer chosen/rejected."""
    return [{"prompt": e["prompt"], "question": e["question"]} for e in examples]
