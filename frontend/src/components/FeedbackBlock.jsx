import { useState } from "react";
import { FiThumbsUp, FiThumbsDown } from "react-icons/fi";
import { REASONS, canProposeCorrection } from "../constants";
import api from "../api";

/**
 * props:
 *   question, answer, sources (list of citation strings), profile
 * Mirrors §5.1 (feedback) + §5.2 (correction, only for avocat_verifie/juriste_interne).
 */
export default function FeedbackBlock({ question, answer, sources, profile }) {
  const [rating, setRating] = useState(null); // "up" | "down" | null
  const [reason, setReason] = useState(null);
  const [reasonText, setReasonText] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [feedbackId, setFeedbackId] = useState(null);
  const [correctionText, setCorrectionText] = useState("");
  const [correctionSubmitted, setCorrectionSubmitted] = useState(false);
  const [error, setError] = useState(null);

  const submitUp = async () => {
    setRating("up");
    try {
      const res = await api.sendFeedback({ question, answer, rating: "up", profile, sources });
      setFeedbackId(res.feedback_id);
      setSubmitted(true);
    } catch (e) {
      setError(e?.response?.data?.detail || "Erreur lors de l'envoi du feedback.");
    }
  };

  const chooseDown = () => setRating("down");

  const submitDown = async () => {
    if (!reason) return;
    try {
      const res = await api.sendFeedback({
        question,
        answer,
        rating: "down",
        profile,
        reason,
        reason_text: reason === "autre" ? reasonText : undefined,
        sources,
      });
      setFeedbackId(res.feedback_id);
      setSubmitted(true);
    } catch (e) {
      setError(e?.response?.data?.detail || "Erreur lors de l'envoi du feedback.");
    }
  };

  const submitCorrection = async () => {
    if (!correctionText.trim() || !feedbackId) return;
    try {
      await api.sendCorrection({
        feedback_id: feedbackId,
        question,
        rejected_answer: answer,
        proposed_answer: correctionText.trim(),
        profile,
      });
      setCorrectionSubmitted(true);
    } catch (e) {
      setError(e?.response?.data?.detail || "Erreur lors de l'envoi de la correction.");
    }
  };

  return (
    <div>
      {!submitted && (
        <div className="feedback-row">
          <span className="fb-note">Cette réponse est-elle utile ?</span>
          <button
            className={`fb-btn up ${rating === "up" ? "active" : ""}`}
            onClick={submitUp}
            aria-label="Utile"
          >
            <FiThumbsUp />
          </button>
          <button
            className={`fb-btn down ${rating === "down" ? "active" : ""}`}
            onClick={chooseDown}
            aria-label="Pas utile"
          >
            <FiThumbsDown />
          </button>
        </div>
      )}

      {rating === "down" && !submitted && (
        <div className="reason-box">
          {REASONS.map((r) => (
            <button
              key={r.value}
              className={`reason-chip ${reason === r.value ? "active" : ""}`}
              onClick={() => setReason(r.value)}
            >
              {r.label}
            </button>
          ))}
          {reason === "autre" && (
            <input
              style={{
                width: "100%",
                marginTop: 8,
                padding: "8px 10px",
                borderRadius: 8,
                border: "1px solid var(--border)",
                fontSize: 13,
              }}
              placeholder="Précisez le motif…"
              value={reasonText}
              onChange={(e) => setReasonText(e.target.value)}
            />
          )}
          {reason && (
            <button className="primary-btn" onClick={submitDown} style={{ width: "100%" }}>
              Envoyer le motif
            </button>
          )}
        </div>
      )}

      {submitted && rating === "up" && (
        <div className="fb-note" style={{ marginTop: 10 }}>
          👍 Merci pour votre retour.
        </div>
      )}

      {submitted && rating === "down" && canProposeCorrection(profile) && !correctionSubmitted && (
        <div className="correction-box">
          <label>Proposer une réponse corrigée (§5.2 — réservé aux profils juridiques)</label>
          <textarea
            value={correctionText}
            onChange={(e) => setCorrectionText(e.target.value)}
            placeholder="Rédigez la réponse correcte…"
          />
          <button className="primary-btn" onClick={submitCorrection} disabled={!correctionText.trim()}>
            Envoyer la correction
          </button>
        </div>
      )}

      {submitted && rating === "down" && !canProposeCorrection(profile) && (
        <div className="fb-note" style={{ marginTop: 10 }}>
          Merci, votre retour a été enregistré.
        </div>
      )}

      {correctionSubmitted && (
        <div className="fb-note" style={{ marginTop: 8 }}>
          ✓ Correction enregistrée — en file d'attente de validation experte.
        </div>
      )}

      {error && <div className="fb-note" style={{ color: "var(--down)", marginTop: 8 }}>{error}</div>}
    </div>
  );
}
