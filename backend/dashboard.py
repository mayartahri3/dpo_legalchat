"""dashboard.py — Métriques de suivi (§5.6), calculées en SQL."""

from __future__ import annotations

from dataclasses import dataclass

from feedback_store import CorrectionStatus as CS, FeedbackStore

ALERT_SATISFACTION_DROP = 0.60
ALERT_NEGATIVE_SPIKE_PER_TOPIC = 5

_OPEN = (CS.PENDING.value, CS.NEEDS_EXPERT.value)
_ACCEPTED = (CS.VALIDATED.value, CS.AUTO_VALIDATED.value, CS.REFORMULATED.value)


@dataclass
class DashboardMetrics:
    total_feedback: int
    satisfaction_rate: float
    pending_corrections: int
    validation_rate: float
    reason_breakdown: dict[str, int]
    style_preference_breakdown: dict[str, int]
    alerts: list[str]


def compute_metrics(store: FeedbackStore) -> DashboardMetrics:
    q = store.conn.execute
    total = store.scalar("SELECT COUNT(*) FROM feedback")
    n_up = store.scalar("SELECT COUNT(*) FROM feedback WHERE rating = 'up'")
    satisfaction = round(n_up / total, 3) if total else 0.0

    reasons = {r["reason"]: r["n"] for r in q(
        "SELECT reason, COUNT(*) AS n FROM feedback WHERE reason IS NOT NULL GROUP BY reason")}

    status_counts = {r["status"]: r["n"] for r in q(
        "SELECT status, COUNT(*) AS n FROM corrections GROUP BY status")}
    pending = sum(status_counts.get(s, 0) for s in _OPEN)
    treated = sum(n for s, n in status_counts.items() if s not in _OPEN)
    accepted = sum(status_counts.get(s, 0) for s in _ACCEPTED)
    validation_rate = round(accepted / treated, 3) if treated else 0.0

    styles = {r["style"]: r["n"] for r in q(
        """SELECT CASE chosen WHEN 'a' THEN style_a ELSE style_b END AS style, COUNT(*) AS n
           FROM style_preferences GROUP BY style""")}

    alerts = []
    if total >= 10 and satisfaction < ALERT_SATISFACTION_DROP:
        alerts.append(f"⚠ Baisse de satisfaction : {satisfaction:.0%} (< {ALERT_SATISFACTION_DROP:.0%}).")
    alerts += [f"⚠ Pic de retours négatifs — motif « {r} » : {n} occurrences."
               for r, n in reasons.items() if n >= ALERT_NEGATIVE_SPIKE_PER_TOPIC]

    return DashboardMetrics(total, satisfaction, pending, validation_rate, reasons, styles, alerts)