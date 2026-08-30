"""
dashboard.py — Suivi et pilotage (§5.6).

Simulation CLI (pas de front dédié ici) : calcule les métriques demandées
par le cahier des charges et affiche les alertes. Les mêmes requêtes
peuvent alimenter un vrai tableau de bord (ex. Streamlit/Grafana) sans
changer cette couche de calcul.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from feedback_store import CorrectionStatus, FeedbackStore, Rating


ALERT_SATISFACTION_DROP = 0.60          # sous ce taux, alerte
ALERT_NEGATIVE_SPIKE_PER_TOPIC = 5      # nb de 👎 sur un même motif -> alerte


@dataclass
class DashboardMetrics:
    total_feedback: int
    satisfaction_rate: float
    pending_corrections: int
    validation_rate: float           # corrections validées / corrections traitées
    reason_breakdown: dict[str, int]
    style_preference_breakdown: dict[str, int]   # §5.5 — style préféré (citoyens/anonymes)
    alerts: list[str]


def compute_metrics(store: FeedbackStore) -> DashboardMetrics:
    fb_rows = store.conn.execute("SELECT rating, reason FROM feedback").fetchall()
    total = len(fb_rows)
    n_up = sum(1 for r in fb_rows if r["rating"] == Rating.UP.value)
    satisfaction_rate = round(n_up / total, 3) if total else 0.0

    reason_counter = Counter(r["reason"] for r in fb_rows if r["reason"])

    corr_rows = store.conn.execute("SELECT status FROM corrections").fetchall()
    pending = sum(1 for r in corr_rows if r["status"] in
                  (CorrectionStatus.PENDING.value, CorrectionStatus.NEEDS_EXPERT.value))
    treated = [r for r in corr_rows if r["status"] not in
               (CorrectionStatus.PENDING.value, CorrectionStatus.NEEDS_EXPERT.value)]
    n_validated = sum(1 for r in treated if r["status"] in
                       (CorrectionStatus.VALIDATED.value, CorrectionStatus.AUTO_VALIDATED.value,
                        CorrectionStatus.REFORMULATED.value))
    validation_rate = round(n_validated / len(treated), 3) if treated else 0.0

    alerts = []
    if total >= 10 and satisfaction_rate < ALERT_SATISFACTION_DROP:
        alerts.append(
            f"⚠ Baisse de satisfaction : {satisfaction_rate:.0%} (< {ALERT_SATISFACTION_DROP:.0%})."
        )
    for reason, count in reason_counter.items():
        if count >= ALERT_NEGATIVE_SPIKE_PER_TOPIC:
            alerts.append(f"⚠ Pic de retours négatifs — motif « {reason} » : {count} occurrences.")

    style_rows = store.conn.execute(
        "SELECT style_a, style_b, chosen FROM style_preferences"
    ).fetchall()
    style_counter = Counter(
        (r["style_a"] if r["chosen"] == "a" else r["style_b"]) for r in style_rows
    )

    return DashboardMetrics(
        total_feedback=total,
        satisfaction_rate=satisfaction_rate,
        pending_corrections=pending,
        validation_rate=validation_rate,
        reason_breakdown=dict(reason_counter),
        style_preference_breakdown=dict(style_counter),
        alerts=alerts,
    )


def print_dashboard(store: FeedbackStore) -> None:
    m = compute_metrics(store)
    print("=" * 56)
    print("TABLEAU DE BORD — Qualité du chatbot juridique")
    print("=" * 56)
    print(f"Feedbacks reçus            : {m.total_feedback}")
    print(f"Taux de satisfaction (👍)   : {m.satisfaction_rate:.0%}")
    print(f"Corrections en attente      : {m.pending_corrections}")
    print(f"Taux de validation          : {m.validation_rate:.0%}")
    print("Motifs d'insatisfaction     :")
    for reason, count in sorted(m.reason_breakdown.items(), key=lambda kv: -kv[1]):
        print(f"  - {reason:<28} {count}")
    if m.style_preference_breakdown:
        print("Style préféré (citoyens)    :")
        for style, count in sorted(m.style_preference_breakdown.items(), key=lambda kv: -kv[1]):
            print(f"  - {style:<28} {count}")
    if m.alerts:
        print("\nALERTES")
        for a in m.alerts:
            print(f"  {a}")
    print("=" * 56)


def model_quality_history(registry_root: str = "models") -> list[dict]:
    """"Évolution de la qualité entre versions du modèle" (§5.6)."""
    from dpo_train import ModelRegistry
    registry = ModelRegistry(registry_root)
    return [
        {
            "version": e["version"],
            "status": e["status"],
            "n_training_pairs": e["n_training_pairs"],
            "eval_summary": e.get("eval_summary", {}),
            "created_at": e["created_at"],
        }
        for e in registry.history()
    ]
