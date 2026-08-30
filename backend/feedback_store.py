"""
feedback_store.py — Collecte du feedback + proposition de correction.

Couvre le cahier des charges :
  §5.1 Collecte du feedback
  §5.2 Proposition de correction
  §6   Traçabilité (question -> réponse -> feedback -> correction -> ...)

Stockage : SQLite (fichier unique, suffisant pour la simulation / le POC ;
remplaçable par Postgres en production sans changer l'API publique de ce
module).
"""

from __future__ import annotations

import enum
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------- #
# Enums — vocabulaire fixé par le cahier des charges
# --------------------------------------------------------------------------- #

class Rating(str, enum.Enum):
    UP = "up"       # "Réponse utile"
    DOWN = "down"   # "Réponse non utile"


class Reason(str, enum.Enum):
    """Motifs prédéfinis d'insatisfaction — §5.1."""
    TOO_GENERAL = "trop_generale"
    LEGALLY_INCORRECT = "juridiquement_incorrecte"
    INCOMPLETE = "incomplete"
    OFF_TOPIC = "hors_sujet"
    OTHER = "autre"


class UserProfile(str, enum.Enum):
    """§4 Acteurs et profils utilisateurs."""
    CITIZEN = "citoyen"
    LAWYER_VERIFIED = "avocat_verifie"
    JURIST_INTERNAL = "juriste_interne"
    ANONYMOUS = "anonyme"


class CorrectionStatus(str, enum.Enum):
    PENDING = "en_attente"          # pas encore vue par le module de validation
    AUTO_VALIDATED = "auto_validee"  # validée automatiquement (confiance + consensus élevés)
    NEEDS_EXPERT = "expert_requis"   # ambiguë / contradictoire / faible confiance
    VALIDATED = "validee"            # validée par un expert
    REJECTED = "rejetee"             # rejetée par un expert
    REFORMULATED = "reformulee"      # reformulée par un expert avant intégration


# Profils autorisés à proposer une correction de contenu juridique (§5.2:
# "Permettre aux utilisateurs qualifiés (avocats, juristes) de proposer une
# correction"). Le citoyen n'a pas la compétence juridique requise — il
# participe autrement, via la préférence de style (§5.5, StylePreference).
CORRECTION_ALLOWED_PROFILES = {UserProfile.LAWYER_VERIFIED, UserProfile.JURIST_INTERNAL}


def can_propose_correction(profile: UserProfile) -> bool:
    return profile in CORRECTION_ALLOWED_PROFILES


class ResponseStyle(str, enum.Enum):
    """§5.5 — styles de réponse entre lesquels un citoyen peut choisir."""
    PEDAGOGICAL = "pedagogique"          # langage simple, explications, exemples
    TECHNICAL = "technique_reference"    # langage technique, références systématiques aux articles


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class Feedback:
    """Une évaluation brute d'une réponse — §5.1."""
    question: str
    answer: str
    rating: Rating
    user_profile: UserProfile = UserProfile.ANONYMOUS
    reason: Reason | None = None            # requis si rating == DOWN
    reason_text: str | None = None          # texte libre si reason == OTHER
    sources: list[str] = field(default_factory=list)  # citations retournées par le RAG
    id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now)


@dataclass
class Correction:
    """Une correction proposée en complément d'un feedback négatif — §5.2."""
    feedback_id: str
    question: str
    rejected_answer: str
    proposed_answer: str
    author_profile: UserProfile
    id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now)
    status: CorrectionStatus = CorrectionStatus.PENDING
    confidence_score: float | None = None
    rag_check: str | None = None            # "confirmee" / "contradiction" / "indeterminee"
    consensus_count: int = 1                # nb d'auteurs indépendants proposant une correction équivalente
    validator_role: str | None = None       # identité/rôle du validateur — §6 Auditabilité
    final_answer: str | None = None         # réponse effectivement retenue (== proposed sauf si reformulée)
    decided_at: str | None = None


@dataclass
class StylePreference:
    """
    Choix « quelle structure de réponse préférez-vous » (façon ChatGPT) —
    §5.5 Personnalisation par profil. Réservé aux profils qui n'ont pas le
    droit de proposer une correction de contenu (citoyen / anonyme) : leur
    contribution porte sur la FORME de la réponse, jamais sur le fond
    juridique, donc aucune validation experte n'est requise ici.
    """
    question: str
    answer_a: str
    style_a: ResponseStyle
    answer_b: str
    style_b: ResponseStyle
    chosen: str                              # "a" ou "b"
    user_profile: UserProfile
    id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating TEXT NOT NULL,
    user_profile TEXT NOT NULL,
    reason TEXT,
    reason_text TEXT,
    sources TEXT,           -- JSON-encoded list
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    feedback_id TEXT NOT NULL REFERENCES feedback(id),
    question TEXT NOT NULL,
    rejected_answer TEXT NOT NULL,
    proposed_answer TEXT NOT NULL,
    author_profile TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence_score REAL,
    rag_check TEXT,
    consensus_count INTEGER NOT NULL DEFAULT 1,
    validator_role TEXT,
    final_answer TEXT,
    timestamp TEXT NOT NULL,
    decided_at TEXT
);

-- Historique des décisions de validation, pour l'auditabilité (§6).
CREATE TABLE IF NOT EXISTS validation_log (
    id TEXT PRIMARY KEY,
    correction_id TEXT NOT NULL REFERENCES corrections(id),
    decision TEXT NOT NULL,       -- validee / rejetee / reformulee / auto_validee
    validator_role TEXT,
    comment TEXT,
    timestamp TEXT NOT NULL
);

-- Préférence de structure/style (§5.5) — réservée aux profils sans droit de
-- correction (citoyen/anonyme). Pas de validation experte : ne porte pas sur
-- le fond juridique, seulement sur la forme.
CREATE TABLE IF NOT EXISTS style_preferences (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer_a TEXT NOT NULL,
    style_a TEXT NOT NULL,
    answer_b TEXT NOT NULL,
    style_b TEXT NOT NULL,
    chosen TEXT NOT NULL,
    user_profile TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
-- Historique de conversation par utilisateur (nouveau) — permet de restaurer
-- le fil de discussion d'un utilisateur après un rafraîchissement de page ou
-- une reconnexion, indépendamment du feedback/correction donné sur chaque tour.
CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    question TEXT NOT NULL,
    mode TEXT NOT NULL,          -- 'single' (avocat/juriste) | 'choice' (citoyen/anonyme)
    answer TEXT,                 -- rempli si mode = 'single'
    answer_a TEXT,               -- rempli si mode = 'choice'
    answer_b TEXT,               -- rempli si mode = 'choice'
    chosen TEXT,                 -- 'a' | 'b' | NULL tant que non choisi
    sources TEXT,                -- JSON-encoded list of {citation, score}
    timestamp TEXT NOT NULL
);
"""


class FeedbackStore:
    """Couche d'accès unique à la base feedback/corrections (backend §3.1)."""

    def __init__(self, db_path: str | Path = "feedback.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- Feedback ----------------------------------------------------------- #

    def add_feedback(self, fb: Feedback) -> Feedback:
        if fb.rating == Rating.DOWN and fb.reason is None:
            raise ValueError("Un motif est requis pour un feedback négatif (§5.1).")
        import json
        self.conn.execute(
            """INSERT INTO feedback
               (id, question, answer, rating, user_profile, reason, reason_text, sources, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fb.id, fb.question, fb.answer, fb.rating.value, fb.user_profile.value,
             fb.reason.value if fb.reason else None, fb.reason_text,
             json.dumps(fb.sources), fb.timestamp),
        )
        self.conn.commit()
        return fb

    def get_feedback(self, feedback_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM feedback WHERE id = ?", (feedback_id,)
        ).fetchone()

    # -- Corrections ---------------------------------------------------------#

    def add_correction(self, c: Correction) -> Correction:
        if not can_propose_correction(c.author_profile):
            raise PermissionError(
                f"Le profil « {c.author_profile.value} » n'est pas autorisé à proposer une "
                f"correction de contenu juridique (§5.2 — réservé à avocat_verifie / "
                f"juriste_interne). Utilisez StylePreference pour ce profil (§5.5)."
            )
        self.conn.execute(
            """INSERT INTO corrections
               (id, feedback_id, question, rejected_answer, proposed_answer, author_profile,
                status, confidence_score, rag_check, consensus_count, validator_role,
                final_answer, timestamp, decided_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (c.id, c.feedback_id, c.question, c.rejected_answer, c.proposed_answer,
             c.author_profile.value, c.status.value, c.confidence_score, c.rag_check,
             c.consensus_count, c.validator_role, c.final_answer, c.timestamp, c.decided_at),
        )
        self.conn.commit()
        return c

    def update_correction(self, c: Correction) -> None:
        self.conn.execute(
            """UPDATE corrections SET
               status=?, confidence_score=?, rag_check=?, consensus_count=?,
               validator_role=?, final_answer=?, decided_at=?
               WHERE id=?""",
            (c.status.value, c.confidence_score, c.rag_check, c.consensus_count,
             c.validator_role, c.final_answer, c.decided_at, c.id),
        )
        self.conn.commit()

    def pending_corrections(self, status: CorrectionStatus | None = None) -> list[sqlite3.Row]:
        if status is None:
            return self.conn.execute(
                "SELECT * FROM corrections WHERE status IN (?, ?)",
                (CorrectionStatus.PENDING.value, CorrectionStatus.NEEDS_EXPERT.value),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM corrections WHERE status = ?", (status.value,)
        ).fetchall()

    def corrections_for_question(self, question: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM corrections WHERE question = ?", (question,)
        ).fetchall()

    def validated_corrections(self) -> list[sqlite3.Row]:
        """Corrections prêtes pour la construction du dataset DPO — §5.4."""
        return self.conn.execute(
            "SELECT * FROM corrections WHERE status IN (?, ?, ?)",
            (CorrectionStatus.VALIDATED.value, CorrectionStatus.AUTO_VALIDATED.value,
             CorrectionStatus.REFORMULATED.value),
        ).fetchall()

    def log_validation(self, correction_id: str, decision: str,
                        validator_role: str | None, comment: str = "") -> None:
        self.conn.execute(
            """INSERT INTO validation_log (id, correction_id, decision, validator_role, comment, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (_new_id(), correction_id, decision, validator_role, comment, _now()),
        )
        self.conn.commit()

    # -- Préférences de style (§5.5) ----------------------------------------- #

    def add_style_preference(self, sp: StylePreference) -> StylePreference:
        if sp.chosen not in {"a", "b"}:
            raise ValueError("chosen doit valoir 'a' ou 'b'.")
        self.conn.execute(
            """INSERT INTO style_preferences
               (id, question, answer_a, style_a, answer_b, style_b, chosen, user_profile, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sp.id, sp.question, sp.answer_a, sp.style_a.value, sp.answer_b, sp.style_b.value,
             sp.chosen, sp.user_profile.value, sp.timestamp),
        )
        self.conn.commit()
        return sp

    def all_style_preferences(self) -> list[sqlite3.Row]:
        """Utilisé par dataset_builder pour construire le dataset de personnalisation (§5.5)."""
        return self.conn.execute("SELECT * FROM style_preferences").fetchall()

    # -- Historique de conversation par utilisateur (nouveau) ----------------- #

    def add_conversation_turn(
        self, user_id: str, profile: "UserProfile | str", question: str, mode: str,
        answer: str | None = None, answer_a: str | None = None, answer_b: str | None = None,
        sources: list | None = None,
    ) -> str:
        import json
        turn_id = _new_id()
        profile_value = profile.value if isinstance(profile, UserProfile) else profile
        self.conn.execute(
            """INSERT INTO conversation_turns
               (id, user_id, profile, question, mode, answer, answer_a, answer_b, chosen, sources, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
            (turn_id, user_id, profile_value, question, mode, answer, answer_a, answer_b,
             json.dumps(sources or []), _now()),
        )
        self.conn.commit()
        return turn_id

    def set_turn_choice(self, turn_id: str, chosen: str) -> None:
        if chosen not in {"a", "b"}:
            raise ValueError("chosen doit valoir 'a' ou 'b'.")
        self.conn.execute("UPDATE conversation_turns SET chosen = ? WHERE id = ?", (chosen, turn_id))
        self.conn.commit()

    def conversation_history(self, user_id: str, limit: int = 200) -> list[sqlite3.Row]:
        """Fil de discussion d'un utilisateur, du plus ancien au plus récent."""
        return self.conn.execute(
            """SELECT * FROM (
                   SELECT * FROM conversation_turns WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT ?
               ) ORDER BY timestamp ASC""",
            (user_id, limit),
        ).fetchall()

    def delete_seeded_demo_data(self) -> int:
        """Supprime les corrections (+ feedback associé) marquées comme données de
        démo (validator_role='seed_demo_data'), pour repartir sur des corrections
        100% réelles avant un entraînement de production."""
        rows = self.conn.execute(
            "SELECT id, feedback_id FROM corrections WHERE validator_role = 'seed_demo_data'"
        ).fetchall()
        correction_ids = [r["id"] for r in rows]
        feedback_ids = [r["feedback_id"] for r in rows]

        if correction_ids:
            self.conn.executemany("DELETE FROM corrections WHERE id = ?", [(i,) for i in correction_ids])
            self.conn.executemany("DELETE FROM validation_log WHERE correction_id = ?", [(i,) for i in correction_ids])
        if feedback_ids:
            self.conn.executemany("DELETE FROM feedback WHERE id = ?", [(i,) for i in feedback_ids])
        self.conn.commit()
        return len(correction_ids)