import { useEffect, useState } from "react";
import api from "../api";

export default function ReviewQueue() {
  const [pending, setPending] = useState(null);
  const [error, setError] = useState(null);
  const [rephraseFor, setRephraseFor] = useState(null);
  const [rephraseText, setRephraseText] = useState("");
  const [validatorRole, setValidatorRole] = useState("expert_juriste");

  const load = () => {
    api
      .reviewPending()
      .then((res) => setPending(res.pending))
      .catch(() => setError("Impossible de charger la file de validation."));
  };

  useEffect(load, []);

  const decide = async (correction_id, decision, reformulated_answer) => {
    try {
      await api.reviewDecide({ correction_id, decision, validator_role: validatorRole, reformulated_answer });
      setRephraseFor(null);
      setRephraseText("");
      load();
    } catch (e) {
      setError(e?.response?.data?.detail || "Erreur lors de la décision.");
    }
  };

  if (error) return <div className="main-scroll"><div className="error-banner">{error}</div></div>;
  if (!pending) return <div className="main-scroll"><div className="empty-state">Chargement…</div></div>;

  return (
    <div className="main-scroll">
      <div className="page">
        <h2>Validation experte</h2>
        <p className="page-sub">
          Corrections ambiguës routées vers un expert juriste (§5.3, §4) — {pending.length} en attente.
        </p>

        <div style={{ marginBottom: 18 }}>
          <label style={{ fontSize: 12.5, fontWeight: 600, color: "var(--text-muted)" }}>
            Rôle du validateur
          </label>
          <input
            style={{
              display: "block",
              marginTop: 6,
              padding: "8px 10px",
              borderRadius: 8,
              border: "1px solid var(--border)",
              fontSize: 13,
              width: 260,
            }}
            value={validatorRole}
            onChange={(e) => setValidatorRole(e.target.value)}
          />
        </div>

        {pending.length === 0 ? (
          <div className="empty-state">
            Rien à faire pour le moment — toutes les corrections sont auto-validées ou traitées.
          </div>
        ) : (
          pending.map((c) => (
            <div className="review-card" key={c.id}>
              <div className="review-meta">
                <span>confiance={c.confidence_score}</span>
                <span>consensus={c.consensus_count}</span>
                <span>RAG={c.rag_check}</span>
                <span>auteur={c.author_profile}</span>
              </div>
              <div><strong>Question :</strong> {c.question}</div>
              <div className="review-answer-label">Rejetée</div>
              <div>{c.rejected_answer}</div>
              <div className="review-answer-label">Proposée</div>
              <div>{c.proposed_answer}</div>

              {rephraseFor === c.id ? (
                <div className="correction-box">
                  <label>Nouvelle réponse reformulée</label>
                  <textarea value={rephraseText} onChange={(e) => setRephraseText(e.target.value)} />
                  <button
                    className="primary-btn"
                    disabled={!rephraseText.trim()}
                    onClick={() => decide(c.id, "reformule", rephraseText.trim())}
                  >
                    Valider la reformulation
                  </button>
                </div>
              ) : (
                <div className="review-actions">
                  <button className="validate" onClick={() => decide(c.id, "valide")}>
                    Valider
                  </button>
                  <button className="reject" onClick={() => decide(c.id, "rejete")}>
                    Rejeter
                  </button>
                  <button className="rephrase" onClick={() => setRephraseFor(c.id)}>
                    Reformuler
                  </button>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
