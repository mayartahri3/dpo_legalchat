import { useEffect, useState } from "react";
import { FiTrendingUp, FiAlertTriangle, FiCheckCircle, FiInfo } from "react-icons/fi";
import api from "../api";

// --------------------------------------------------------------------------- #
// Moteur de recommandations — purement dérivé des données déjà exposées par
// l'API (dashboard, train/stats, models/history, train/jobs, health). Pas de
// nouvel endpoint : on combine ce qui existe pour donner des conclusions
// actionnables plutôt que de simples chiffres bruts.
// --------------------------------------------------------------------------- #

function buildRecommendations({ metrics, trainStats, history, jobs, health }) {
  const recs = [];
  const push = (level, text) => recs.push({ level, text });

  // --- Données de démo encore en base ---
  if (trainStats?.demo_corrections > 0) {
    push(
      "warning",
      `${trainStats.demo_corrections} correction(s) de démo sont encore en base — supprimez-les avant tout entraînement de production (page Entraînement DPO).`
    );
  }

  // --- File de validation experte ---
  if (metrics?.pending_corrections > 5) {
    push(
      "warning",
      `${metrics.pending_corrections} correction(s) en attente de validation experte — la file est chargée, priorisez la revue pour ne pas retarder le prochain cycle DPO.`
    );
  } else if (metrics?.pending_corrections > 0) {
    push("info", `${metrics.pending_corrections} correction(s) en attente de validation experte.`);
  }

  // --- Progression vers le seuil d'auto-entraînement ---
  if (trainStats) {
    const { validated_corrections_real: real, auto_train_threshold: threshold } = trainStats;
    const runningJob = jobs?.some((j) => ["queued", "running"].includes(j.status));
    if (real >= threshold && !runningJob) {
      push(
        "success",
        `Seuil atteint (${real}/${threshold} corrections validées réelles) — un nouveau cycle DPO peut être lancé.`
      );
    } else if (real < threshold) {
      push(
        "info",
        `${threshold - real} correction(s) validée(s) supplémentaire(s) requise(s) avant le déclenchement automatique (${real}/${threshold}).`
      );
    }
  }

  // --- Version entraînée mais jamais promue ---
  const stagedVersions = (history || []).filter((v) => v.status === "staged");
  if (stagedVersions.length > 0) {
    const latest = stagedVersions[stagedVersions.length - 1];
    push(
      "warning",
      `La version v${latest.version} (${latest.n_training_pairs} paires) est entraînée mais pas encore promue — évaluez-la avant mise en production.`
    );
  }

  // --- Aucune version fine-tunée active ---
  const activeVersion = (history || []).find((v) => v.status === "active");
  if (!activeVersion && history?.length > 0) {
    push(
      "info",
      "Aucune version fine-tunée n'est promue en production — le modèle de base (non personnalisé) est actuellement utilisé."
    );
  }
  if (!activeVersion && (!history || history.length === 0)) {
    push(
      "info",
      "Aucun cycle d'entraînement n'a encore été exécuté — le chatbot tourne sur le modèle de base uniquement."
    );
  }

  // --- Sur-apprentissage potentiel ---
  if (activeVersion && activeVersion.n_training_pairs > 0 && activeVersion.n_training_pairs < 15) {
    push(
      "warning",
      `La version active (v${activeVersion.version}) a été entraînée sur seulement ${activeVersion.n_training_pairs} exemples — risque de sur-apprentissage, surveillez la qualité des réponses de près.`
    );
  }

  // --- Satisfaction globale ---
  if (metrics && metrics.total_feedback >= 5) {
    const pct = Math.round(metrics.satisfaction_rate * 100);
    if (pct < 60) {
      push("warning", `Taux de satisfaction faible (${pct}%) — examinez les motifs d'insatisfaction ci-dessous.`);
    } else if (pct >= 85) {
      push("success", `Taux de satisfaction élevé (${pct}%) sur ${metrics.total_feedback} retours.`);
    }
  } else if (metrics && metrics.total_feedback < 5) {
    push("info", "Peu de retours utilisateurs pour l'instant — les métriques de satisfaction ne sont pas encore fiables.");
  }

  // --- Motif d'insatisfaction dominant ---
  const reasonEntries = Object.entries(metrics?.reason_breakdown || {});
  if (reasonEntries.length > 0) {
    const totalReasons = reasonEntries.reduce((s, [, c]) => s + c, 0);
    const [topReason, topCount] = reasonEntries.sort((a, b) => b[1] - a[1])[0];
    const share = topCount / totalReasons;
    if (share >= 0.4 && totalReasons >= 5) {
      push(
        "info",
        `Le motif d'insatisfaction dominant est « ${topReason} » (${Math.round(share * 100)}% des retours négatifs) — envisagez d'ajuster le prompt système ou le top_k de récupération en conséquence.`
      );
    }
  }

  // --- Préférence de style dominante (citoyens/anonymes) ---
  const styleEntries = Object.entries(metrics?.style_preference_breakdown || {});
  if (styleEntries.length > 0) {
    const totalStyles = styleEntries.reduce((s, [, c]) => s + c, 0);
    const [topStyle, topCount] = styleEntries.sort((a, b) => b[1] - a[1])[0];
    const share = topCount / totalStyles;
    if (share >= 0.7 && totalStyles >= 5) {
      push(
        "info",
        `Les citoyens préfèrent nettement le style « ${topStyle} » (${Math.round(share * 100)}% des choix) — envisagez de l'utiliser comme réponse unique par défaut plutôt que de proposer les deux variantes.`
      );
    }
  }

  // --- Dernier job d'entraînement en échec ---
  const lastJob = jobs?.[0]; // trainJobs() renvoie les plus récents en premier
  if (lastJob?.status === "failed") {
    push("warning", `Le dernier cycle d'entraînement a échoué : ${lastJob.error || "erreur inconnue"}.`);
  }

  // --- n8n configuré mais inactif, ou l'inverse ---
  if (trainStats?.auto_train_via_n8n && !trainStats?.n8n_configured) {
    push("warning", "L'auto-entraînement via n8n est activé mais aucun webhook n'est configuré — les cycles automatiques ne partiront pas.");
  }

  // --- Santé générale du pipeline ---
  if (health && !health.loaded_model_path) {
    push("warning", "Impossible de déterminer quel modèle est actuellement chargé par l'API.");
  }

  return recs;
}

const LEVEL_ICON = { warning: FiAlertTriangle, success: FiCheckCircle, info: FiInfo };

function RecommendationCard({ level, text }) {
  const Icon = LEVEL_ICON[level] || FiInfo;
  return (
    <div className={`rec-card rec-${level}`}>
      <Icon className="rec-icon" />
      <span>{text}</span>
    </div>
  );
}

function Bar({ value, max, colorVar = "--accent" }) {
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0;
  return (
    <div className="bar-track">
      <div className="bar-fill" style={{ width: `${pct}%`, background: `var(${colorVar})` }} />
    </div>
  );
}

const STATUS_LABEL = { active: "Active", staged: "En attente", retired: "Retirée" };

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [trainStats, setTrainStats] = useState(null);
  const [history, setHistory] = useState(null);
  const [jobs, setJobs] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  const load = () => {
    Promise.all([
      api.dashboard(),
      api.trainStats().catch(() => null),
      api.modelsHistory().catch(() => ({ history: [] })),
      api.trainJobs().catch(() => ({ jobs: [] })),
      api.health().catch(() => null),
    ])
      .then(([m, ts, mh, tj, h]) => {
        setMetrics(m);
        setTrainStats(ts);
        setHistory(mh.history || []);
        setJobs(tj.jobs || []);
        setHealth(h);
      })
      .catch(() => setError("Impossible de charger le tableau de bord."));
  };

  useEffect(load, []);

  if (error) return <div className="main-scroll"><div className="error-banner">{error}</div></div>;
  if (!metrics) return <div className="main-scroll"><div className="empty-state">Chargement…</div></div>;

  const recommendations = buildRecommendations({ metrics, trainStats, history, jobs, health });
  const activeVersion = (history || []).find((v) => v.status === "active");
  const maxReason = Math.max(1, ...Object.values(metrics.reason_breakdown || {}));
  const maxStyle = Math.max(1, ...Object.values(metrics.style_preference_breakdown || {}));

  // Entonnoir du pipeline : feedback -> retours négatifs -> corrections
  // proposées -> validées -> utilisées par la version active.
  const downVotes = Math.round(metrics.total_feedback * (1 - metrics.satisfaction_rate));
  const funnelSteps = [
    { label: "Feedbacks reçus", value: metrics.total_feedback },
    { label: "Retours négatifs 👎", value: downVotes },
    { label: "Corrections proposées", value: trainStats?.total_corrections ?? 0 },
    { label: "Corrections validées", value: trainStats?.validated_corrections ?? 0 },
    { label: "Utilisées (version active)", value: activeVersion?.n_training_pairs ?? 0 },
  ];
  const funnelMax = Math.max(1, ...funnelSteps.map((s) => s.value));

  return (
    <div className="main-scroll">
      <div className="page page-wide">
        <h2>Tableau de bord</h2>
        <p className="page-sub">Suivi de la qualité du chatbot juridique et de la boucle d'amélioration (§5.6).</p>

        {/* --- KPIs principaux --- */}
        <div className="metric-grid metric-grid-6">
          <div className="metric-card">
            <div className="metric-value">{metrics.total_feedback}</div>
            <div className="metric-label">Feedbacks reçus</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{Math.round(metrics.satisfaction_rate * 100)}%</div>
            <div className="metric-label">Taux de satisfaction 👍</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.pending_corrections}</div>
            <div className="metric-label">Corrections en attente</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{Math.round(metrics.validation_rate * 100)}%</div>
            <div className="metric-label">Taux de validation</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">
              {trainStats ? `${trainStats.validated_corrections_real}/${trainStats.auto_train_threshold}` : "—"}
            </div>
            <div className="metric-label">Progression vers le prochain cycle</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{activeVersion ? `v${activeVersion.version}` : "Base"}</div>
            <div className="metric-label">Modèle en production</div>
          </div>
        </div>

        {/* --- Recommandations --- */}
        <div className="section-title">
          <FiTrendingUp style={{ verticalAlign: -2, marginRight: 6 }} />
          Recommandations
        </div>
        {recommendations.length === 0 ? (
          <div className="empty-state">Rien à signaler pour le moment — le pipeline fonctionne normalement.</div>
        ) : (
          <div className="rec-list">
            {recommendations.map((r, i) => (
              <RecommendationCard key={i} level={r.level} text={r.text} />
            ))}
          </div>
        )}

        {/* --- Entonnoir du pipeline --- */}
        <div className="section-title">Entonnoir de la boucle d'amélioration</div>
        <div className="funnel">
          {funnelSteps.map((s, i) => (
            <div className="funnel-step" key={i}>
              <div className="funnel-step-header">
                <span>{s.label}</span>
                <span className="funnel-step-value">{s.value}</span>
              </div>
              <Bar value={s.value} max={funnelMax} />
            </div>
          ))}
        </div>

        {/* --- Historique des versions --- */}
        <div className="section-title">Historique des versions du modèle</div>
        {(!history || history.length === 0) ? (
          <div className="empty-state">Aucun cycle d'entraînement exécuté pour le moment.</div>
        ) : (
          history
            .slice()
            .reverse()
            .map((v) => (
              <div className="review-card" key={v.version}>
                <div className="review-meta">
                  <span className={`version-badge version-${v.status}`}>{STATUS_LABEL[v.status] || v.status}</span>
                  <span>v{v.version}</span>
                  <span>{v.n_training_pairs} paires d'entraînement</span>
                  <span>{v.created_at ? new Date(v.created_at).toLocaleString("fr-FR") : ""}</span>
                </div>
              </div>
            ))
        )}

        {/* --- Jobs d'entraînement récents --- */}
        {jobs && jobs.length > 0 && (
          <>
            <div className="section-title">Cycles d'entraînement récents</div>
            {jobs.slice(0, 5).map((j) => (
              <div className="review-card" key={j.job_id}>
                <div className="review-meta">
                  <span className={`version-badge version-${j.status === "completed" ? "active" : j.status === "failed" ? "retired" : "staged"}`}>
                    {j.status}
                  </span>
                  <span>déclenché : {j.trigger === "auto" ? "automatiquement" : "manuellement"}</span>
                  {j.n_training_pairs != null && <span>{j.n_training_pairs} paires</span>}
                </div>
                {j.status === "failed" && <div className="fb-note" style={{ color: "var(--down)" }}>{j.error}</div>}
              </div>
            ))}
          </>
        )}

        {/* --- Répartitions --- */}
        <div className="section-title">Motifs d'insatisfaction</div>
        {Object.keys(metrics.reason_breakdown).length === 0 ? (
          <div className="empty-state">Aucun motif enregistré pour le moment.</div>
        ) : (
          Object.entries(metrics.reason_breakdown)
            .sort((a, b) => b[1] - a[1])
            .map(([reason, count]) => (
              <div className="breakdown-row-bar" key={reason}>
                <div className="breakdown-row">
                  <span>{reason}</span>
                  <span>{count}</span>
                </div>
                <Bar value={count} max={maxReason} colorVar="--down" />
              </div>
            ))
        )}

        <div className="section-title">Style préféré (citoyens / anonymes)</div>
        {Object.keys(metrics.style_preference_breakdown).length === 0 ? (
          <div className="empty-state">Aucune préférence de style enregistrée pour le moment.</div>
        ) : (
          Object.entries(metrics.style_preference_breakdown)
            .sort((a, b) => b[1] - a[1])
            .map(([style, count]) => (
              <div className="breakdown-row-bar" key={style}>
                <div className="breakdown-row">
                  <span>{style}</span>
                  <span>{count}</span>
                </div>
                <Bar value={count} max={maxStyle} colorVar="--accent" />
              </div>
            ))
        )}

        {metrics.alerts.length > 0 && (
          <>
            <div className="section-title">Alertes système</div>
            {metrics.alerts.map((a, i) => (
              <div className="alert-item" key={i}>
                {a}
              </div>
            ))}
          </>
        )}

        {/* --- Santé du système --- */}
        {health && (
          <>
            <div className="section-title">État du système</div>
            <div className="review-card">
              <div className="review-meta" style={{ flexWrap: "wrap" }}>
                <span>Modèle chargé : {health.loaded_model_path || "?"}</span>
                <span>Articles : {health.articles_dir}</span>
                <span>top_k : {health.top_k}</span>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}