"""
model_eval.py — comparaison de versions pour l'UI (/api/models/compare) et colab_eval.py.
Même signature qu'avant ; la logique de métriques est dans eval_metrics.py.

Évalue sur le HOLDOUT si eval_split.json existe (sinon sur toutes les paires validées,
ce qui mesure de la mémorisation — le champ `split` du résultat l'indique).
"""

from __future__ import annotations

import numpy as np

from dataset_builder import build_preference_dataset, load_holdout_questions
from eval_metrics import article_numbers, cited_numbers, evaluate_model
from feedback_store import FeedbackStore
from train_common import load_tokenizer


def compare_model_versions(store: FeedbackStore, index, versions: list[dict], base_model: str,
                           top_k: int = 3, n_questions: int = 15, max_new_tokens: int = 300,
                           beta: float = 0.1) -> dict:
    split = "test" if load_holdout_questions() else "all"
    examples = build_preference_dataset(store, index, load_tokenizer(base_model),
                                        min_examples=1, top_k=top_k, split=split)[:n_questions]
    if not examples:
        raise ValueError("Aucune question validée disponible pour la comparaison.")

    ctx_nums = []
    for ex in examples:
        ctx = set()
        for a, _ in index.search(ex["question"], top_k=top_k):
            ctx |= article_numbers(a)
        ctx_nums.append(ctx)

    # Un modèle chargé à la fois ; "base" sert de référence pour la marge DPO implicite.
    candidates = [("base", base_model)] + [(f"v{v['version']}", v["path"]) for v in versions]
    res = {label: evaluate_model(path, base_model, examples, ctx_nums, index.model, max_new_tokens)
           for label, path in candidates}

    ref = res["base"]
    labels = [f"v{v['version']}" for v in versions]
    sims = np.stack([res[l]["emb_sim"] for l in labels])
    winners = sims.argmax(axis=0)   # win-rate parmi les versions comparées (base exclue)

    out = []
    for i, v in enumerate(versions):
        r = res[labels[i]]
        margin = (r["lp_chosen"] - ref["lp_chosen"]) - (r["lp_rejected"] - ref["lp_rejected"])
        out.append({
            "version": v["version"], "method": v.get("method", "?"), "path": v["path"],
            "mean_similarity": round(float(r["emb_sim"].mean()), 4),
            "win_rate": round(float((winners == i).mean()), 4),
            "n_questions": len(examples),
            "pref_acc": round(float((beta * margin > 0).mean()), 4),
            "margin": round(float(margin.mean()), 4),
            "rouge_l": round(float(r["rouge_l"].mean()), 4),
            "halluc_rate": round(float(r["halluc"].mean()), 4),
        })
    return {"n_questions": len(examples), "split": split, "results": out}