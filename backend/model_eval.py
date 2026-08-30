"""
model_eval.py — Comparaison DPO vs PPO vs GRPO (et vs le modèle de base) sur
un jeu de questions commun, avec des métriques identiques pour les trois.

Deux métriques, calculées pour chaque version comparée :
  1. similarité moyenne à la réponse validée ("chosen") — cosine similarity
     d'embedding, même principe que ReferenceRewardScorer (dpo_train.py).
     Une valeur entre 0 et 1 ; plus haut = plus proche de ce qu'un
     avocat/juriste a validé comme bonne réponse.
  2. taux de victoire (win-rate) par comparaison par paires : pour chaque
     question, quelle version produit la réponse la plus proche de la
     réponse validée ? Le win-rate d'une version = part des questions où
     elle bat toutes les autres versions comparées.

Utilisation typique : comparer la version active DPO, la dernière version
PPO et la dernière version GRPO sur les mêmes questions validées, pour
décider laquelle promouvoir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset_builder import build_preference_dataset
from feedback_store import FeedbackStore
from minirag import EmbeddingIndex, LocalLLM


@dataclass
class VersionResult:
    version: int
    method: str
    path: str
    mean_similarity: float
    win_rate: float
    n_questions: int


def compare_model_versions(store: FeedbackStore, index: EmbeddingIndex,
                            versions: list[dict], base_model: str,
                            top_k: int = 3, n_questions: int = 15,
                            max_new_tokens: int = 300) -> dict:
    """
    `versions` : liste d'entrées ModelRegistry.history() (dicts avec au moins
    version/method/path) à comparer entre elles. Charge chaque modèle,
    génère une réponse pour les mêmes `n_questions` questions validées, et
    calcule les métriques ci-dessus.

    ATTENTION : charge chaque modèle en mémoire l'un après l'autre (pas tous
    simultanément) pour limiter l'usage RAM/VRAM — donc plus il y a de
    versions à comparer, plus cette fonction est lente. Prévoir plusieurs
    minutes pour 3 versions sur CPU.
    """
    tokenizer_probe = LocalLLM(base_model, max_new_tokens=1)  # juste pour obtenir un tokenizer cohérent
    examples = build_preference_dataset(store, index, tokenizer_probe.tokenizer, min_examples=1, top_k=top_k)
    del tokenizer_probe
    examples = examples[:n_questions]
    if not examples:
        raise ValueError("Aucune question validée disponible pour la comparaison.")

    embed_model = index.model
    generations_by_version: dict[int, list[str]] = {}

    for v in versions:
        llm = LocalLLM(v["path"], max_new_tokens=max_new_tokens, base_model_for_adapter=base_model)
        outputs = []
        for ex in examples:
            hits = index.search(ex["question"], top_k=top_k)
            articles = [a for a, _score in hits]
            outputs.append(llm.generate(ex["question"], articles))
        generations_by_version[v["version"]] = outputs
        del llm  # libère la mémoire avant de charger la version suivante

    # Similarité moyenne par version
    results: dict[int, dict] = {}
    per_question_scores: dict[int, list[float]] = {}
    for v in versions:
        scores = []
        for ex, gen in zip(examples, generations_by_version[v["version"]]):
            vecs = embed_model.encode([gen, ex["chosen"]], normalize_embeddings=True)
            scores.append(float(np.dot(vecs[0], vecs[1])))
        per_question_scores[v["version"]] = scores
        results[v["version"]] = {
            "version": v["version"],
            "method": v.get("method", "?"),
            "path": v["path"],
            "mean_similarity": round(float(np.mean(scores)), 4),
            "n_questions": len(examples),
        }

    # Win-rate par paires : pour chaque question, qui a le meilleur score ?
    wins = {v["version"]: 0 for v in versions}
    for qi in range(len(examples)):
        best_version, best_score = None, -1.0
        for v in versions:
            score = per_question_scores[v["version"]][qi]
            if score > best_score:
                best_score, best_version = score, v["version"]
        wins[best_version] += 1
    for v in versions:
        results[v["version"]]["win_rate"] = round(wins[v["version"]] / len(examples), 4)

    return {
        "n_questions": len(examples),
        "results": list(results.values()),
    }
