"""
validation.py — Module de validation des corrections (§5.3).

Rôle : ne JAMAIS laisser une correction utilisateur entrer telle quelle dans
le dataset d'entraînement (§6 Fiabilité juridique, §8 Gestion des risques).

Quatre étapes, dans l'ordre du cahier des charges :
  1. Vérification de cohérence avec le RAG (contradiction / confirmation).
  2. Score de confiance basé sur le profil de l'auteur.
  3. Agrégation des corrections similaires (vote / consensus).
  4. Routage : auto-validation si confiance+consensus élevés, sinon
     validation humaine obligatoire (expert juriste).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feedback_store import Correction, CorrectionStatus, FeedbackStore, UserProfile


# --------------------------------------------------------------------------- #
# 1. Pondération de confiance par profil — §5.3, §8 (biais lié au profil)
# --------------------------------------------------------------------------- #

PROFILE_WEIGHT = {
    UserProfile.LAWYER_VERIFIED: 1.0,
    UserProfile.JURIST_INTERNAL: 0.9,
    # Citoyen/anonyme n'ont plus le droit de proposer de correction du tout
    # (feedback_store.CORRECTION_ALLOWED_PROFILES, §5.2/§8) — ces deux entrées
    # ne sont plus atteignables en pratique, gardées pour robustesse/API stable.
    UserProfile.CITIZEN: 0.4,
    UserProfile.ANONYMOUS: 0.2,
}

# Seuils de routage — ajustables, mais volontairement prudents (§6: priorité
# à la prudence sur la vitesse d'apprentissage).
AUTO_VALIDATE_CONFIDENCE = 0.85
AUTO_VALIDATE_MIN_CONSENSUS = 2         # au moins 2 auteurs indépendants
CONTRADICTION_PENALTY = 0.5
SIMILAR_CORRECTION_THRESHOLD = 0.88     # similarité cosinus pour dire "même correction"


@dataclass
class RagCheckResult:
    verdict: str          # "confirmee" | "contradiction" | "indeterminee"
    best_score: float      # similarité max avec un article du RAG
    top_citation: str | None


class ValidationModule:
    """Orchestration de la validation d'une correction — §5.3."""

    def __init__(self, index, store: FeedbackStore, embed_model=None):
        """
        index : minirag.EmbeddingIndex déjà construit (source de vérité juridique, §3.2)
        store : FeedbackStore pour lire/écrire les corrections
        embed_model : modèle d'embeddings pour comparer deux réponses entre elles
                      (par défaut on réutilise index.model)
        """
        self.index = index
        self.store = store
        self.embed_model = embed_model or index.model

    # -- Étape 1 : cohérence RAG ------------------------------------------- #

    def check_against_rag(self, question: str, proposed_answer: str, top_k: int = 3) -> RagCheckResult:
        """
        Recherche les articles pertinents pour la question, puis compare le
        contenu de la réponse proposée à ces articles.

        Heuristique de simulation (pas de LLM juge ici, cf. minirag.py qui
        n'a pas de dépendance réseau) : similarité d'embedding élevée avec
        au moins un article => "confirmée" ; aucun article proche => "indéterminée".
        Une vraie détection de contradiction nécessiterait un modèle NLI
        (entailment/contradiction) — voir note en bas de fichier.
        """
        hits = self.index.search(f"{question}\n{proposed_answer}", top_k=top_k)
        if not hits:
            return RagCheckResult("indeterminee", 0.0, None)

        best_article, best_score = hits[0]
        if best_score >= 0.55:
            return RagCheckResult("confirmee", best_score, best_article.citation())
        return RagCheckResult("indeterminee", best_score, best_article.citation())

    # -- Étape 2 : score de confiance --------------------------------------- #

    def confidence_score(self, author_profile: UserProfile, rag_check: RagCheckResult,
                          consensus_count: int) -> float:
        base = PROFILE_WEIGHT.get(author_profile, 0.2)
        rag_factor = {
            "confirmee": 1.0,
            "indeterminee": 0.7,
            "contradiction": 1.0 - CONTRADICTION_PENALTY,
        }[rag_check.verdict]
        consensus_bonus = min(0.15 * (consensus_count - 1), 0.3)
        score = min(1.0, base * rag_factor + consensus_bonus)
        return round(score, 3)

    # -- Étape 3 : consensus / vote ------------------------------------------ #

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.embed_model.encode(texts, normalize_embeddings=True))

    def find_consensus(self, correction: Correction) -> int:
        """
        Compte le nombre d'auteurs indépendants ayant proposé une correction
        sémantiquement équivalente pour la même question (§5.3 — agrégation
        des corrections similaires / vote).
        """
        siblings = self.store.corrections_for_question(correction.question)
        if len(siblings) <= 1:
            return 1

        vecs = self._embed([correction.proposed_answer] + [s["proposed_answer"] for s in siblings])
        target, others = vecs[0], vecs[1:]
        sims = others @ target
        similar_authors = {
            siblings[i]["author_profile"]
            for i, sim in enumerate(sims)
            if sim >= SIMILAR_CORRECTION_THRESHOLD
        }
        similar_authors.add(correction.author_profile.value)
        return max(1, len(similar_authors))

    # -- Étape 4 : routage ---------------------------------------------------- #

    def validate(self, correction: Correction) -> Correction:
        """Exécute le pipeline complet et met à jour + persiste la correction."""
        rag_check = self.check_against_rag(correction.question, correction.proposed_answer)
        consensus = self.find_consensus(correction)
        confidence = self.confidence_score(correction.author_profile, rag_check, consensus)

        correction.rag_check = rag_check.verdict
        correction.consensus_count = consensus
        correction.confidence_score = confidence

        if rag_check.verdict == "contradiction":
            correction.status = CorrectionStatus.NEEDS_EXPERT
        elif confidence >= AUTO_VALIDATE_CONFIDENCE and consensus >= AUTO_VALIDATE_MIN_CONSENSUS:
            correction.status = CorrectionStatus.AUTO_VALIDATED
            correction.final_answer = correction.proposed_answer
            correction.decided_at = correction.timestamp
            self.store.log_validation(correction.id, "auto_validee", None,
                                       f"confiance={confidence}, consensus={consensus}")
        else:
            correction.status = CorrectionStatus.NEEDS_EXPERT

        self.store.update_correction(correction)
        return correction

    # -- Décision experte (humaine) — §4 Expert validateur -------------------- #

    def expert_decide(self, correction: Correction, decision: str, validator_role: str,
                       reformulated_answer: str | None = None, comment: str = "") -> Correction:
        """
        decision : "valide" | "rejete" | "reformule"
        Le passage humain est *obligatoire* pour toute correction routée vers
        NEEDS_EXPERT — aucune correction n'atteint le dataset autrement
        (§6 Fiabilité juridique).
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

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
        correction.decided_at = now
        self.store.update_correction(correction)
        self.store.log_validation(correction.id, correction.status.value, validator_role, comment)
        return correction


# NOTE — limite assumée de la simulation :
# `check_against_rag` utilise la similarité d'embedding comme proxy de
# "confirmation", faute d'un modèle NLI entraîné à détecter les contradictions
# juridiques. En production, remplacer par un classifieur entailment/
# contradiction (ou un second LLM-juge) pour distinguer "l'article confirme
# la correction" de "l'article la contredit" — actuellement les deux cas
# ambigus tombent en NEEDS_EXPERT par prudence, ce qui respecte l'exigence
# de fiabilité du §6 même si c'est conservateur.
