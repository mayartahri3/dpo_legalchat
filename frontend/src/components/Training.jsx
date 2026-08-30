import { useEffect, useState } from "react";
import { FiCpu, FiZap, FiGitMerge } from "react-icons/fi";
import api from "../api";

const STATUS_LABEL = { active: "Active", staged: "En attente", retired: "Retirée" };
const JOB_STATUS_LABEL = {
  queued: "En file",
  running: "En cours",
  completed: "Terminé",
  failed: "Échec",
  skipped: "Ignoré",
};
const METHOD_LABEL = { dpo: "DPO", ppo: "PPO", grpo: "GRPO", reward_model: "Reward Model" };
const METHODS = [
  { value: "dpo", label: "DPO", hint: "Léger — pas de reward model requis." },
  { value: "ppo", label: "PPO", hint: "Nécessite un reward model entraîné au préalable." },
  { value: "grpo", label: "GRPO", hint: "Pas de reward model requis — récompense par similarité." },
];

export default function Training() {
  const [stats, setStats] = useState(null);
  const [minExamples, setMinExamples] = useState(20);
  const [epochs, setEpochs] = useState(1);
  const [method, setMethod] = useState("dpo");
  const [jobs, setJobs] = useState([]);
  const [history, setHistory] = useState([]);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [error, setError] = useState(null);
  const [n8nTriggering, setN8nTriggering] = useState(false);
  const [selectedForCompare, setSelectedForCompare] = useState([]);
  const [comparing, setComparing] = useState(false);
  const [comparison, setComparison] = useState(null);

  const loadStats = () => {
    api
      .trainStats()
      .then((res) => {
        setStats(res);
        setMinExamples(res.auto_train_threshold);
      })
      .catch(() => {});
  };

  const loadJobs = () => {
    api
      .trainJobs()
      .then((res) => setJobs(res.jobs))
      .catch(() => {});
  };

  const loadHistory = () => {
    api
      .modelsHistory()
      .then((res) => setHistory(res.history))
      .catch(() => {});
  };

  const loadRewardHistory = () => {
    api
      .rewardModelHistory()
      .then((res) => setRewardHistory(res.history))
      .catch(() => {});
  };

  useEffect(() => {
    loadStats();
    loadJobs();
    loadHistory();
    loadRewardHistory();
  }, []);

  // Poll en continu : montre aussi bien un job lancé manuellement qu'un job
  // déclenché automatiquement par un autre onglet/navigateur au moment où le
  // seuil de corrections validées a été atteint.
  useEffect(() => {
    const t = setInterval(() => {
      loadStats();
      loadJobs();
    }, 4000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const runningJob = jobs.find((j) => ["queued", "running"].includes(j.status));
    if (!runningJob) {
      loadHistory();
      loadRewardHistory();
    }
  }, [jobs]);

  const launch = async () => {
    setError(null);
    try {
      await api.train({ min_examples: minExamples, epochs, method });
      loadJobs();
    } catch (e) {
      setError(e?.response?.data?.detail || "Erreur au lancement de l'entraînement.");
    }
  };

  const toggleCompareSelection = (version) => {
    setSelectedForCompare((sel) =>
      sel.includes(version) ? sel.filter((v) => v !== version) : [...sel, version]
    );
  };

  const runComparison = async () => {
    if (selectedForCompare.length < 2) {
      setError("Sélectionnez au moins deux versions à comparer.");
      return;
    }
    setError(null);
    setComparing(true);
    setComparison(null);
    try {
      const res = await api.compareModels({ versions: selectedForCompare });
      setComparison(res);
    } catch (e) {
      setError(e?.response?.data?.detail || "Erreur lors de la comparaison.");
    } finally {
      setComparing(false);
    }
  };

  const triggerViaN8n = async () => {
    setError(null);
    setN8nTriggering(true);
    try {
      await api.triggerN8n({ min_examples: minExamples, epochs });
    } catch (e) {
      setError(e?.response?.data?.detail || "Erreur lors du déclenchement via n8n.");
    } finally {
      setN8nTriggering(false);
    }
  };

  const promote = async (version) => {
    try {
      await api.promote(version);
      loadHistory();
    } catch (e) {
      setError(e?.response?.data?.detail || "Erreur lors de la promotion.");
    }
  };

  const runningJob = jobs.find((j) => ["queued", "running"].includes(j.status));
  const realCount = stats?.validated_corrections_real ?? 0;
  const threshold = stats?.auto_train_threshold ?? 20;
  const progressPct = Math.min(100, Math.round((realCount / threshold) * 100));

  return (
    <div className="main-scroll">
      <div className="page page-wide">
        <h2>
          <FiCpu style={{ verticalAlign: -3, marginRight: 8 }} />
          Entraînement DPO
        </h2>
        <p className="page-sub">
          Cycle réel (dataset des corrections validées → <code>trl.DPOTrainer</code>, LoRA) via{" "}
          <code>dpo_train.py</code>, en tâche de fond — §5.4, §7. Basé sur les corrections
          issues d'un usage réel (chat → feedback → correction → validation).
        </p>

        {/* --- Progression vers le prochain cycle --- */}
        <div className="review-card">
          <div style={{ fontSize: 13.5, fontWeight: 600, marginBottom: 10 }}>
            Corrections validées disponibles pour l'entraînement
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 8 }}>
            <span style={{ fontSize: 28, fontWeight: 700 }}>{realCount}</span>
            <span style={{ color: "var(--text-muted)", fontSize: 13.5 }}>
              / {threshold} — seuil de déclenchement automatique
            </span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${progressPct}%`, background: "var(--accent)" }} />
          </div>
          <div className="fb-note" style={{ marginTop: 10 }}>
            {realCount >= threshold
              ? "Seuil atteint — un cycle se déclenche automatiquement dès qu'une nouvelle correction est validée (voir §5.3 → Validation experte)."
              : `Encore ${threshold - realCount} correction(s) validée(s) requise(s) pour le déclenchement automatique. Utilisez le chat (profil avocat/juriste) et la page Validation experte pour en ajouter.`}
          </div>
          {stats && (
            <div className="fb-note" style={{ marginTop: 8 }}>
              Déclenchement automatique via {stats.auto_train_via_n8n ? "n8n" : "cycle local"}
              {stats.auto_train_via_n8n && !stats.n8n_configured && (
                <span style={{ color: "var(--down)" }}> — ⚠ aucun webhook n8n configuré, les cycles auto ne partiront pas.</span>
              )}
            </div>
          )}
        </div>

        {/* --- Choix de la méthode --- */}
        <div className="section-title" style={{ marginTop: 18 }}>Méthode d'entraînement</div>
        <div className="method-selector">
          {METHODS.map((m) => (
            <button
              key={m.value}
              className={`method-option ${method === m.value ? "active" : ""}`}
              onClick={() => setMethod(m.value)}
              type="button"
            >
              <div className="method-option-label">{m.label}</div>
              <div className="method-option-hint">{m.hint}</div>
            </button>
          ))}
        </div>

        {method === "ppo" && (
          <div className="fb-note" style={{ margin: "6px 0 4px" }}>
            {rewardHistory.length === 0 ? (
              <>
                ⚠ Aucun reward model entraîné — PPO ne peut pas démarrer sans lui.{" "}
                <button
                  className="primary-btn"
                  style={{ marginTop: 6, padding: "6px 12px" }}
                  onClick={async () => {
                    setError(null);
                    try {
                      await api.train({ min_examples: minExamples, epochs, method: "reward_model" });
                      loadJobs();
                    } catch (e) {
                      setError(e?.response?.data?.detail || "Erreur au lancement de l'entraînement du reward model.");
                    }
                  }}
                >
                  Entraîner un reward model d'abord
                </button>
              </>
            ) : (
              `✓ Reward model disponible (v${rewardHistory[rewardHistory.length - 1].version}, ${rewardHistory[rewardHistory.length - 1].n_training_pairs} paires).`
            )}
          </div>
        )}

        {/* --- Lancement manuel --- */}
        <div style={{ display: "flex", gap: 14, marginBottom: 10, alignItems: "flex-end", marginTop: 10, flexWrap: "wrap" }}>
          <div>
            <label style={{ fontSize: 12.5, fontWeight: 600, color: "var(--text-muted)" }}>
              Min. corrections validées
            </label>
            <input
              type="number"
              min={1}
              value={minExamples}
              onChange={(e) => setMinExamples(Number(e.target.value))}
              style={{
                display: "block", marginTop: 6, padding: "8px 10px", borderRadius: 8,
                border: "1px solid var(--border)", fontSize: 13, width: 140,
              }}
            />
          </div>
          <div>
            <label style={{ fontSize: 12.5, fontWeight: 600, color: "var(--text-muted)" }}>
              Époques
            </label>
            <input
              type="number"
              min={1}
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              style={{
                display: "block", marginTop: 6, padding: "8px 10px", borderRadius: 8,
                border: "1px solid var(--border)", fontSize: 13, width: 100,
              }}
            />
          </div>
          <button
            className="primary-btn"
            onClick={launch}
            disabled={!!runningJob || (method === "ppo" && rewardHistory.length === 0)}
            style={{ marginTop: 0 }}
          >
            {runningJob ? "Entraînement en cours…" : `Lancer un cycle ${METHOD_LABEL[method]}`}
          </button>
          {stats?.n8n_configured && method === "dpo" && (
            <button
              className="primary-btn"
              onClick={triggerViaN8n}
              disabled={n8nTriggering}
              style={{ marginTop: 0, background: "white", color: "var(--accent)", border: "1px solid var(--accent)" }}
            >
              <FiZap style={{ verticalAlign: -2, marginRight: 4 }} />
              {n8nTriggering ? "Envoi…" : "Déclencher via n8n"}
            </button>
          )}
        </div>
        <p className="page-sub" style={{ marginTop: -4 }}>
          Vous pouvez lancer manuellement avec un petit nombre (ex. 2-3) pour tester, même sous
          le seuil d'auto-déclenchement — le champ ci-dessus n'est pas limité au seuil.
        </p>

        {error && <div className="error-banner">{error}</div>}

        {jobs.length > 0 && (
          <>
            <div className="section-title">Jobs d'entraînement</div>
            {jobs.map((job) => (
              <div className="review-card" key={job.job_id}>
                <div className="review-meta">
                  <span
                    className={`version-badge version-${
                      job.status === "completed" ? "active" : job.status === "failed" ? "retired" : "staged"
                    }`}
                  >
                    {JOB_STATUS_LABEL[job.status] || job.status}
                  </span>
                  <span>déclenché : {job.trigger === "auto" ? "automatiquement" : "manuellement"}</span>
                </div>
                {job.status === "running" && (
                  <div className="loading-dots"><span /><span /><span /></div>
                )}
                {job.status === "skipped" && <div className="fb-note">{job.message}</div>}
                {job.status === "completed" && (
                  <div className="fb-note">
                    ✓ Version v{job.version} entraînée ({job.n_training_pairs} paires) — statut{" "}
                    <strong>{job.model_status}</strong>. À évaluer avant promotion.
                  </div>
                )}
                {job.status === "failed" && (
                  <div className="fb-note" style={{ color: "var(--down)" }}>Échec : {job.error}</div>
                )}
              </div>
            ))}
          </>
        )}

        <div className="section-title">Historique des versions</div>
        {history.length === 0 ? (
          <div className="empty-state">Aucun cycle d'entraînement encore exécuté.</div>
        ) : (
          <>
            <p className="page-sub" style={{ marginTop: -4 }}>
              Cochez au moins deux versions pour les comparer (similarité à la réponse validée + win-rate) sur les mêmes questions.
            </p>
            {history
              .slice()
              .reverse()
              .map((v) => (
                <div className="review-card" key={v.version}>
                  <div className="review-meta" style={{ alignItems: "center" }}>
                    <input
                      type="checkbox"
                      checked={selectedForCompare.includes(v.version)}
                      onChange={() => toggleCompareSelection(v.version)}
                    />
                    <span className={`version-badge version-${v.status}`}>{STATUS_LABEL[v.status] || v.status}</span>
                    <span className="method-badge">{METHOD_LABEL[v.method] || v.method || "dpo"}</span>
                    <span>v{v.version}</span>
                    <span>{v.n_training_pairs} paires</span>
                    <span>{v.created_at ? new Date(v.created_at).toLocaleString("fr-FR") : ""}</span>
                  </div>
                  {v.eval_summary?.mean_similarity != null && (
                    <div className="fb-note" style={{ marginTop: 6 }}>
                      Dernière évaluation : similarité {Math.round(v.eval_summary.mean_similarity * 100)}% ·
                      win-rate {Math.round(v.eval_summary.win_rate * 100)}% (sur {v.eval_summary.n_questions} questions)
                    </div>
                  )}
                  {v.status === "staged" && (
                    <button className="primary-btn" onClick={() => promote(v.version)}>
                      Promouvoir en production
                    </button>
                  )}
                </div>
              ))}

            <button
              className="primary-btn"
              onClick={runComparison}
              disabled={comparing || selectedForCompare.length < 2}
              style={{ marginTop: 10 }}
            >
              <FiGitMerge style={{ verticalAlign: -2, marginRight: 6 }} />
              {comparing ? "Comparaison en cours (peut prendre plusieurs minutes)…" : `Comparer ${selectedForCompare.length} version(s)`}
            </button>

            {comparison && (
              <div className="review-card" style={{ marginTop: 14 }}>
                <div style={{ fontSize: 13.5, fontWeight: 600, marginBottom: 10 }}>
                  Résultats de comparaison ({comparison.n_questions} questions)
                </div>
                <table className="compare-table">
                  <thead>
                    <tr>
                      <th>Version</th>
                      <th>Méthode</th>
                      <th>Similarité moyenne</th>
                      <th>Win-rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.results
                      .slice()
                      .sort((a, b) => b.mean_similarity - a.mean_similarity)
                      .map((r) => (
                        <tr key={r.version}>
                          <td>v{r.version}</td>
                          <td><span className="method-badge">{METHOD_LABEL[r.method] || r.method}</span></td>
                          <td>{Math.round(r.mean_similarity * 100)}%</td>
                          <td>{Math.round(r.win_rate * 100)}%</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
                <p className="fb-note" style={{ marginTop: 10 }}>
                  Similarité = proximité (embedding) entre la réponse générée et la réponse validée par un expert.
                  Win-rate = part des questions où cette version bat toutes les autres comparées.
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
