"""
validation.py — Validation des corrections (§5.3) : jamais d'entrée directe dans le
dataset. Cohérence RAG -> confiance par profil -> consensus -> routage (auto / expert).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from feedback_store import Correction, CorrectionStatus, FeedbackStore, UserProfile

PROFILE_WEIGHT = {
    UserProfile.LAWYER_VERIFIED: 1.0,
    UserProfile.JURIST_INTERNAL: 0.9,
    UserProfile.CITIZEN: 0.4,       # n'ont plus le droit de corriger (§5.2) — gardés pour l'API stable
    UserProfile.ANONYMOUS: 0.2,
}
RAG_FACTOR = {"confirmee": 1.0, "indeterminee": 0.7, "contradiction": 0.5}

AUTO_VALIDATE_CONFIDENCE = 0.85
AUTO_VALIDATE_MIN_CONSENSUS = 2
SIMILAR_CORRECTION_THRESHOLD = 0.88
RAG_CONFIRM_THRESHOLD = 0.55


@dataclass
class RagCheckResult:
    verdict: str            # "confirmee" | "contradiction" | "indeterminee"
    best_score: float
    top_citation: str | None


class ValidationModule:
    def __init__(self, index, store: FeedbackStore, embed_model=None):
        self.index = index
        self.store = store
        self.embed_model = embed_model or index.model

    def check_against_rag(self, question: str, proposed_answer: str, top_k: int = 3) -> RagCheckResult:
        """Proxy par similarité d'embedding (pas de modèle NLI) : détecter une vraie
        contradiction nécessiterait un classifieur entailment/contradiction. Le cas
        ambigu tombe en NEEDS_EXPERT, ce qui est conservateur (§6)."""
        hits = self.index.search(f"{question}\n{proposed_answer}", top_k=top_k)
        if not hits:
            return RagCheckResult("indeterminee", 0.0, None)
        article, score = hits[0]
        verdict = "confirmee" if score >= RAG_CONFIRM_THRESHOLD else "indeterminee"
        return RagCheckResult(verdict, score, article.citation())

    @staticmethod
    def confidence_score(author_profile: UserProfile, rag_check: RagCheckResult,
                         consensus_count: int) -> float:
        base = PROFILE_WEIGHT.get(author_profile, 0.2)
        bonus = min(0.15 * (consensus_count - 1), 0.3)
        return round(min(1.0, base * RAG_FACTOR[rag_check.verdict] + bonus), 3)

    def find_consensus(self, correction: Correction) -> int:
        """Nombre de profils d'auteurs distincts proposant une correction équivalente."""
        siblings = self.store.corrections_for_question(correction.question)
        if len(siblings) <= 1:
            return 1
        texts = [correction.proposed_answer] + [s["proposed_answer"] for s in siblings]
        vecs = np.asarray(self.embed_model.encode(texts, normalize_embeddings=True))
        sims = vecs[1:] @ vecs[0]
        authors = {s["author_profile"] for s, sim in zip(siblings, sims)
                   if sim >= SIMILAR_CORRECTION_THRESHOLD}
        authors.add(correction.author_profile.value)
        return len(authors)

    def validate(self, correction: Correction) -> Correction:
        rag = self.check_against_rag(correction.question, correction.proposed_answer)
        consensus = self.find_consensus(correction)
        confidence = self.confidence_score(correction.author_profile, rag, consensus)

        correction.rag_check = rag.verdict
        correction.consensus_count = consensus
        correction.confidence_score = confidence

        auto = (rag.verdict != "contradiction" and confidence >= AUTO_VALIDATE_CONFIDENCE
                and consensus >= AUTO_VALIDATE_MIN_CONSENSUS)
        if auto:
            correction.status = CorrectionStatus.AUTO_VALIDATED
            correction.final_answer = correction.proposed_answer
            correction.decided_at = correction.timestamp
            self.store.log_validation(correction.id, "auto_validee", None,
                                      f"confiance={confidence}, consensus={consensus}")
        else:
            correction.status = CorrectionStatus.NEEDS_EXPERT

        self.store.update_correction(correction)
        return correction

    def expert_decide(self, correction: Correction, decision: str, validator_role: str,
                      reformulated_answer: str | None = None, comment: str = "") -> Correction:
        """decision : "valide" | "rejete" | "reformule" — passage humain obligatoire (§6)."""
        if decision == "valide":
            correction.status = CorrectionStatus.VALIDATED
            correction.final_answer = correction.proposed_answer
        elif decision == "reformule":
            if not reformulated_answer:
                raise ValueError("Une réponse reformulée est requise pour ce type de décision.")
            correction.status = CorrectionStatus.REFORMULATED
            correction.final_answer = reformulated_answer
        elif decision == "rejete":
            correction.status = CorrectionStatus.REJECTED
            correction.final_answer = None
        else:
            raise ValueError(f"Décision inconnue : {decision}")

        correction.validator_role = validator_role
        correction.decided_at = datetime.now(timezone.utc).isoformat()
        self.store.update_correction(correction)
        self.store.log_validation(correction.id, correction.status.value, validator_role, comment)
        return correction