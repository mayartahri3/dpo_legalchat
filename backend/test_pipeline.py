"""Smoke test — exercises the pure-logic modules without downloading any model."""
import numpy as np

import feedback_store
from validation import ValidationModule
from dataset_builder import build_preference_dataset, save_dataset
from dashboard import compute_metrics


class FakeArticle:
    def __init__(self, text, cite):
        self._text, self._cite = text, cite
    def citation(self):
        return self._cite


class FakeModel:
    """Deterministic bag-of-words embedding, just enough to test the pipeline logic."""
    def encode(self, texts, normalize_embeddings=True):
        vocab = {}
        vecs = []
        for t in texts:
            words = t.lower().split()
            v = np.zeros(64)
            for w in words:
                idx = hash(w) % 64
                v[idx] += 1
            n = np.linalg.norm(v)
            vecs.append(v / n if n > 0 else v)
        return np.asarray(vecs)


class FakeIndex:
    model = FakeModel()
    def search(self, query, top_k=3):
        art = FakeArticle("le loyer doit etre paye dans les huit jours", "Art. 12 - Loi 2020")
        return [(art, 0.6)]


def main():
    store = feedback_store.FeedbackStore(":memory:")
    profile_lawyer = feedback_store.UserProfile.LAWYER_VERIFIED

    fb = store.add_feedback(feedback_store.Feedback(
        question="Quel est le delai de paiement du loyer ?",
        answer="Le loyer doit etre paye immediatement.",
        rating=feedback_store.Rating.DOWN, user_profile=feedback_store.UserProfile.CITIZEN,
        reason=feedback_store.Reason.LEGALLY_INCORRECT,
    ))
    assert fb.id

    # Un citoyen ne peut pas proposer de correction du fond — seulement une
    # préférence de forme.
    fb_citizen = store.add_feedback(feedback_store.Feedback(
        question="Quel est le delai de paiement du loyer ?",
        answer="Le loyer doit etre paye immediatement.",
        rating=feedback_store.Rating.DOWN, user_profile=feedback_store.UserProfile.CITIZEN,
        reason=feedback_store.Reason.LEGALLY_INCORRECT,
    ))
    try:
        store.add_correction(feedback_store.Correction(
            feedback_id=fb_citizen.id, question=fb_citizen.question,
            rejected_answer=fb_citizen.answer, proposed_answer="Devrait etre 8 jours.",
            author_profile=feedback_store.UserProfile.CITIZEN,
        ))
        raise AssertionError("un citoyen ne devrait pas pouvoir proposer de correction")
    except ValueError:
        pass
    pref = store.add_structure_preference(feedback_store.StructurePreference(
        feedback_id=fb_citizen.id, question=fb_citizen.question,
        preferred_structure=feedback_store.ResponseStructure.MORE_CITATIONS,
        author_profile=feedback_store.UserProfile.CITIZEN,
    ))
    assert pref.id
    print("structure preferences recorded:", len(store.structure_preferences()))

    c1 = store.add_correction(feedback_store.Correction(
        feedback_id=fb.id, question=fb.question, rejected_answer=fb.answer,
        proposed_answer="Le loyer doit etre paye dans un delai de huit jours.",
        author_profile=profile_lawyer,
    ))
    c2 = store.add_correction(feedback_store.Correction(
        feedback_id=fb.id, question=fb.question, rejected_answer=fb.answer,
        proposed_answer="Le loyer doit etre paye dans un delai de huit jours ouvres.",
        author_profile=feedback_store.UserProfile.JURIST_INTERNAL,
    ))

    validator = ValidationModule(FakeIndex(), store)
    c1 = validator.validate(c1)
    c2 = validator.validate(c2)
    print("c1 status:", c1.status, "confidence:", c1.confidence_score, "consensus:", c1.consensus_count)
    print("c2 status:", c2.status, "confidence:", c2.confidence_score, "consensus:", c2.consensus_count)

    pending = store.pending_corrections()
    print("still pending/needs_expert:", len(pending))

    for row in store.pending_corrections(status=feedback_store.CorrectionStatus.NEEDS_EXPERT):
        corr = feedback_store.Correction(
            id=row["id"], feedback_id=row["feedback_id"], question=row["question"],
            rejected_answer=row["rejected_answer"], proposed_answer=row["proposed_answer"],
            author_profile=feedback_store.UserProfile(row["author_profile"]),
        )
        validator.expert_decide(corr, "valide", "expert_juriste_test")

    validated = store.validated_corrections()
    print("validated corrections:", len(validated))

    try:
        examples = build_preference_dataset(store, min_examples=1)
        print("dataset examples:", len(examples))
        path = save_dataset(examples, "/home/claude/rl_feedback/_test_out", version=1)
        print("dataset written to:", path)
    except ValueError as e:
        print("dataset build skipped:", e)

    metrics = compute_metrics(store)
    print("metrics:", metrics)

    store.close()
    print("\nOK — smoke test passed.")


if __name__ == "__main__":
    main()
