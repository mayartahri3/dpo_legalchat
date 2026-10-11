"""
feedback_store.py — Feedback (§5.1), corrections (§5.2), préférences de style
(§5.5), traçabilité (§6) et historique de conversation. SQLite.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class Rating(str, enum.Enum):
    UP = "up"
    DOWN = "down"


class Reason(str, enum.Enum):
    TOO_GENERAL = "trop_generale"
    LEGALLY_INCORRECT = "juridiquement_incorrecte"
    INCOMPLETE = "incomplete"
    OFF_TOPIC = "hors_sujet"
    OTHER = "autre"


class UserProfile(str, enum.Enum):
    CITIZEN = "citoyen"
    LAWYER_VERIFIED = "avocat_verifie"
    JURIST_INTERNAL = "juriste_interne"
    ANONYMOUS = "anonyme"


class CorrectionStatus(str, enum.Enum):
    PENDING = "en_attente"
    AUTO_VALIDATED = "auto_validee"
    NEEDS_EXPERT = "expert_requis"
    VALIDATED = "validee"
    REJECTED = "rejetee"
    REFORMULATED = "reformulee"


class ResponseStyle(str, enum.Enum):
    PEDAGOGICAL = "pedagogique"
    TECHNICAL = "technique_reference"


# Seuls ces profils peuvent corriger le fond juridique (§5.2) ; les autres
# n'influencent que la FORME (StylePreference, §5.5).
CORRECTION_ALLOWED_PROFILES = {UserProfile.LAWYER_VERIFIED, UserProfile.JURIST_INTERNAL}


def can_propose_correction(profile: UserProfile) -> bool:
    return profile in CORRECTION_ALLOWED_PROFILES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Data classes (les noms de champs == noms de colonnes SQL)
# --------------------------------------------------------------------------- #

@dataclass
class Feedback:
    question: str
    answer: str
    rating: Rating
    user_profile: UserProfile = UserProfile.ANONYMOUS
    reason: Reason | None = None
    reason_text: str | None = None
    sources: list[str] = field(default_factory=list)
    id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now)


@dataclass
class Correction:
    feedback_id: str
    question: str
    rejected_answer: str
    proposed_answer: str
    author_profile: UserProfile
    id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now)
    status: CorrectionStatus = CorrectionStatus.PENDING
    confidence_score: float | None = None
    rag_check: str | None = None
    consensus_count: int = 1
    validator_role: str | None = None
    final_answer: str | None = None
    decided_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Correction":
        d = dict(row)
        d["author_profile"] = UserProfile(d["author_profile"])
        d["status"] = CorrectionStatus(d["status"])
        return cls(**d)


@dataclass
class StylePreference:
    question: str
    answer_a: str
    style_a: ResponseStyle
    answer_b: str
    style_b: ResponseStyle
    chosen: str
    user_profile: UserProfile
    id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now)


def _row(obj) -> dict:
    """dataclass -> dict prêt pour sqlite (enums -> valeur, listes -> JSON)."""
    out = {}
    for k, v in dataclasses.asdict(obj).items():
        out[k] = v.value if isinstance(v, enum.Enum) else json.dumps(v) if isinstance(v, list) else v
    return out


# --------------------------------------------------------------------------- #
# Schéma
# --------------------------------------------------------------------------- #

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL, rating TEXT NOT NULL,
    user_profile TEXT NOT NULL, reason TEXT, reason_text TEXT, sources TEXT, timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY, feedback_id TEXT NOT NULL REFERENCES feedback(id),
    question TEXT NOT NULL, rejected_answer TEXT NOT NULL, proposed_answer TEXT NOT NULL,
    author_profile TEXT NOT NULL, status TEXT NOT NULL, confidence_score REAL, rag_check TEXT,
    consensus_count INTEGER NOT NULL DEFAULT 1, validator_role TEXT, final_answer TEXT,
    timestamp TEXT NOT NULL, decided_at TEXT
);
CREATE TABLE IF NOT EXISTS validation_log (
    id TEXT PRIMARY KEY, correction_id TEXT NOT NULL REFERENCES corrections(id),
    decision TEXT NOT NULL, validator_role TEXT, comment TEXT, timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS style_preferences (
    id TEXT PRIMARY KEY, question TEXT NOT NULL, answer_a TEXT NOT NULL, style_a TEXT NOT NULL,
    answer_b TEXT NOT NULL, style_b TEXT NOT NULL, chosen TEXT NOT NULL,
    user_profile TEXT NOT NULL, timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, profile TEXT NOT NULL, question TEXT NOT NULL,
    mode TEXT NOT NULL, answer TEXT, answer_a TEXT, answer_b TEXT, chosen TEXT,
    sources TEXT, timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corr_status   ON corrections(status);
CREATE INDEX IF NOT EXISTS idx_corr_question ON corrections(question);
CREATE INDEX IF NOT EXISTS idx_turns_user    ON conversation_turns(user_id, timestamp);
"""

_INITIALIZED: set[str] = set()   # chemins de DB déjà initialisés dans ce process


class FeedbackStore:
    """Accès unique à la base. Une instance par requête (voir api.get_store)."""

    def __init__(self, db_path: str | Path = "feedback.db"):
        key = str(db_path)
        self.db_path = Path(db_path)
        # check_same_thread=False : FastAPI peut exécuter la dépendance et la route
        # dans deux threads différents ; la connexion n'est jamais utilisée en parallèle.
        self.conn = sqlite3.connect(key, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if key == ":memory:" or key not in _INITIALIZED:
            self.conn.executescript(SCHEMA)
            _INITIALIZED.add(key)

    def close(self) -> None:
        self.conn.close()

    def scalar(self, sql: str, params: tuple = ()):
        return self.conn.execute(sql, params).fetchone()[0]

    def _insert(self, table: str, row: dict) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
        self.conn.commit()

    # -- Feedback ----------------------------------------------------------- #

    def add_feedback(self, fb: Feedback) -> Feedback:
        if fb.rating == Rating.DOWN and fb.reason is None:
            raise ValueError("Un motif est requis pour un feedback négatif (§5.1).")
        self._insert("feedback", _row(fb))
        return fb

    def get_feedback(self, feedback_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,)).fetchone()

    # -- Corrections -------------------------------------------------------- #

    def add_correction(self, c: Correction) -> Correction:
        if not can_propose_correction(c.author_profile):
            raise PermissionError(
                f"Le profil « {c.author_profile.value} » n'est pas autorisé à proposer une "
                "correction de contenu juridique (§5.2 — réservé à avocat_verifie / juriste_interne)."
            )
        self._insert("corrections", _row(c))
        return c

    def update_correction(self, c: Correction) -> None:
        self.conn.execute(
            """UPDATE corrections SET status=?, confidence_score=?, rag_check=?, consensus_count=?,
               validator_role=?, final_answer=?, decided_at=? WHERE id=?""",
            (c.status.value, c.confidence_score, c.rag_check, c.consensus_count,
             c.validator_role, c.final_answer, c.decided_at, c.id),
        )
        self.conn.commit()

    def get_correction(self, correction_id: str) -> Correction | None:
        row = self.conn.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,)).fetchone()
        return Correction.from_row(row) if row else None

    def pending_corrections(self, status: CorrectionStatus | None = None) -> list[sqlite3.Row]:
        statuses = ((status.value,) if status
                    else (CorrectionStatus.PENDING.value, CorrectionStatus.NEEDS_EXPERT.value))
        marks = ", ".join("?" * len(statuses))
        return self.conn.execute(f"SELECT * FROM corrections WHERE status IN ({marks})", statuses).fetchall()

    def corrections_for_question(self, question: str) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM corrections WHERE question = ?", (question,)).fetchall()

    def validated_corrections(self) -> list[sqlite3.Row]:
        """Corrections prêtes pour le dataset — §5.4."""
        return self.conn.execute(
            "SELECT * FROM corrections WHERE status IN (?, ?, ?)",
            (CorrectionStatus.VALIDATED.value, CorrectionStatus.AUTO_VALIDATED.value,
             CorrectionStatus.REFORMULATED.value),
        ).fetchall()

    def log_validation(self, correction_id: str, decision: str,
                       validator_role: str | None, comment: str = "") -> None:
        self._insert("validation_log", {
            "id": _new_id(), "correction_id": correction_id, "decision": decision,
            "validator_role": validator_role, "comment": comment, "timestamp": _now(),
        })

    # -- Préférences de style ----------------------------------------------- #

    def add_style_preference(self, sp: StylePreference) -> StylePreference:
        if sp.chosen not in {"a", "b"}:
            raise ValueError("chosen doit valoir 'a' ou 'b'.")
        self._insert("style_preferences", _row(sp))
        return sp

    def all_style_preferences(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM style_preferences").fetchall()

    # -- Historique de conversation ----------------------------------------- #

    def add_conversation_turn(self, user_id: str, profile: "UserProfile | str", question: str, mode: str,
                              answer: str | None = None, answer_a: str | None = None,
                              answer_b: str | None = None, sources: list | None = None) -> str:
        turn_id = _new_id()
        self._insert("conversation_turns", {
            "id": turn_id, "user_id": user_id,
            "profile": profile.value if isinstance(profile, UserProfile) else profile,
            "question": question, "mode": mode, "answer": answer, "answer_a": answer_a,
            "answer_b": answer_b, "chosen": None, "sources": json.dumps(sources or []),
            "timestamp": _now(),
        })
        return turn_id

    def set_turn_choice(self, turn_id: str, chosen: str) -> None:
        if chosen not in {"a", "b"}:
            raise ValueError("chosen doit valoir 'a' ou 'b'.")
        self.conn.execute("UPDATE conversation_turns SET chosen = ? WHERE id = ?", (chosen, turn_id))
        self.conn.commit()

    def conversation_history(self, user_id: str, limit: int = 200) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT * FROM (SELECT * FROM conversation_turns WHERE user_id = ?
                              ORDER BY timestamp DESC LIMIT ?) ORDER BY timestamp ASC""",
            (user_id, limit),
        ).fetchall()

    # -- Maintenance --------------------------------------------------------- #

    def delete_seeded_demo_data(self) -> int:
        rows = self.conn.execute(
            "SELECT id, feedback_id FROM corrections WHERE validator_role = 'seed_demo_data'"
        ).fetchall()
        if not rows:
            return 0
        c_ids = [r["id"] for r in rows]
        f_ids = [r["feedback_id"] for r in rows]
        c_marks, f_marks = ", ".join("?" * len(c_ids)), ", ".join("?" * len(f_ids))
        self.conn.execute(f"DELETE FROM validation_log WHERE correction_id IN ({c_marks})", c_ids)
        self.conn.execute(f"DELETE FROM corrections WHERE id IN ({c_marks})", c_ids)
        self.conn.execute(f"DELETE FROM feedback WHERE id IN ({f_marks})", f_ids)
        self.conn.commit()
        return len(c_ids)