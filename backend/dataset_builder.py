"""
dataset_builder.py — Constitution du dataset de préférences (§5.4) + split train/test.

  - split="train" (défaut) : exclut les questions du holdout (eval_split.json) s'il existe.
  - split="test"           : ne garde QUE le holdout.
  - split="all"            : tout.
eval_split.json doit aussi être présent sur Colab (voir export_for_colab.py).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from feedback_store import FeedbackStore
from minirag import EmbeddingIndex, build_chat_prompt


SYSTEM_PROMPT = (
    "Tu es un assistant juridique. Réponds UNIQUEMENT à partir des articles "
    "de loi fournis en contexte. Si le contexte ne contient pas la réponse, "
    "dis-le clairement. Cite systématiquement le numéro d'article utilisé."
)

HOLDOUT_PATH = Path(__file__).resolve().parent / "eval_split.json"


def make_holdout_split(store: FeedbackStore, test_frac: float = 0.25, seed: int = 42,
                       path: str | Path = HOLDOUT_PATH) -> dict:
    """Réserve une fraction des QUESTIONS (pas des lignes) pour l'évaluation."""
    questions = sorted({r["question"] for r in store.validated_corrections()})
    if len(questions) < 4:
        raise ValueError(
            f"Seulement {len(questions)} question(s) validée(s) — il en faut au moins 4 "
            "pour avoir un split train/test minimal."
        )
    random.Random(seed).shuffle(questions)
    n_test = min(max(2, round(len(questions) * test_frac)), len(questions) - 2)

    split = {
        "seed": seed,
        "test_frac": test_frac,
        "n_questions_total": len(questions),
        "test_questions": sorted(questions[:n_test]),
    }
    Path(path).write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    return split


def load_holdout_questions(path: str | Path = HOLDOUT_PATH) -> set[str] | None:
    p = Path(path)
    if not p.exists():
        return None
    return set(json.loads(p.read_text(encoding="utf-8")).get("test_questions", []))


def build_preference_dataset(store: FeedbackStore, index: EmbeddingIndex, tokenizer,
                             min_examples: int = 1, top_k: int = 3,
                             system_prompt: str = SYSTEM_PROMPT,
                             split: str = "train") -> list[dict]:
    """Liste de dicts {"prompt", "chosen", "rejected", "question", "correction_id", "confidence_score"}."""
    if split not in {"train", "test", "all"}:
        raise ValueError("split doit valoir 'train', 'test' ou 'all'.")

    rows = store.validated_corrections()
    holdout = load_holdout_questions()

    if split == "test":
        if holdout is None:
            raise ValueError("eval_split.json introuvable — lancez d'abord "
                             "`python eval_metrics.py --make-split`.")
        rows = [r for r in rows if r["question"] in holdout]
    elif split == "train" and holdout:
        n_before = len(rows)
        rows = [r for r in rows if r["question"] not in holdout]
        print(f"[dataset_builder] split=train : {n_before - len(rows)} ligne(s) du holdout exclue(s).")

    examples = []
    for row in rows:
        chosen = row["final_answer"] or row["proposed_answer"]
        rejected = row["rejected_answer"]
        if not chosen or not rejected or chosen.strip() == rejected.strip():
            continue  # paire dégénérée

        hits = index.search(row["question"], top_k=top_k)
        prompt = build_chat_prompt(tokenizer, row["question"], [a for a, _ in hits], system_prompt)
        examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "question": row["question"],
            "correction_id": row["id"],
            "confidence_score": row["confidence_score"],
        })

    if len(examples) < min_examples:
        raise ValueError(
            f"Seulement {len(examples)} paire(s) validée(s) pour split='{split}' "
            f"(minimum requis: {min_examples}). Voir §8 — risque de sur-apprentissage."
        )
    return examples


def save_dataset(examples: list[dict], out_dir: str | Path, version: int) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dataset_v{version}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    (out_dir / f"dataset_v{version}.meta.json").write_text(json.dumps({
        "version": version,
        "n_examples": len(examples),
        "correction_ids": [ex["correction_id"] for ex in examples],
    }, indent=2), encoding="utf-8")
    return path


def training_columns(examples: list[dict]) -> list[dict]:
    return [{"prompt": e["prompt"], "chosen": e["chosen"], "rejected": e["rejected"]} for e in examples]


def prompt_pool(examples: list[dict]) -> list[dict]:
    return [{"prompt": e["prompt"], "question": e["question"]} for e in examples]