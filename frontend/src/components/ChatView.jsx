import { useEffect, useRef, useState } from "react";
import {
  FiSend,
  FiPlus,
  FiPaperclip,
  FiBookOpen,
  FiFileText,
  FiBriefcase,
  FiCompass,
} from "react-icons/fi";
import api from "../api";
import FeedbackBlock from "./FeedbackBlock";
import { getUserId } from "../user";
import { upsertConversation, titleFromTurns } from "../conversations";

const SUGGESTIONS = [
  {
    icon: FiCompass,
    title: "Expliquer une loi",
    desc: "Explique la loi relative au télétravail.",
  },
  {
    icon: FiFileText,
    title: "Comprendre une procédure",
    desc: "Comment enregistrer une marque ?",
  },
  {
    icon: FiBriefcase,
    title: "Droit du travail",
    desc: "Quels sont les droits d'un salarié pendant sa période d'essai ?",
  },
  {
    icon: FiBookOpen,
    title: "Créer et gérer une entreprise",
    desc: "Quelles sont les étapes pour créer une société ?",
  },
];

function SourcesList({ sources }) {
  if (!sources?.length) return null;
  return (
    <div className="sources">
      {sources.map((s, i) => (
        <div className="source-chip" key={i}>
          <span className="source-score">({s.score})</span>
          {s.citation}
        </div>
      ))}
    </div>
  );
}

function SmallTalkAnswer({ turn }) {
  return (
    <div className="msg-bubble" style={{ width: "100%" }}>
      <div className="answer-card">{turn.answer}</div>
    </div>
  );
}

function SingleAnswer({ turn, profile }) {
  return (
    <div className="msg-bubble" style={{ width: "100%" }}>
      <div className="answer-card">
        <span className="answer-label">Réponse</span>
        <div>{turn.answer}</div>
        <SourcesList sources={turn.sources} />
      </div>
      <FeedbackBlock
        question={turn.question}
        answer={turn.answer}
        sources={(turn.sources || []).map((s) => s.citation)}
        profile={profile}
      />
    </div>
  );
}

function ChoiceAnswer({ turn, profile, onChosen }) {
  const [chosen, setChosen] = useState(turn.chosen || null);
  const [saving, setSaving] = useState(false);

  const choose = async (which) => {
    if (chosen || saving) return;
    setSaving(true);
    try {
      await api.sendStylePreference({
        question: turn.question,
        answer_a: turn.answer_a,
        answer_b: turn.answer_b,
        chosen: which,
        profile,
        turn_id: turn.turn_id,
      });
      setChosen(which);
      onChosen(which);
    } finally {
      setSaving(false);
    }
  };

  // Une fois un style choisi, on n'affiche plus QUE la réponse retenue —
  // l'autre variante disparaît de la conversation. Le choix reste enregistré
  // (dataset de personnalisation §5.5) mais la lecture reste simple pour
  // l'utilisateur : une seule réponse à lire, pas deux à comparer en continu.
  if (chosen) {
    const chosenText = chosen === "a" ? turn.answer_a : turn.answer_b;
    const label = chosen === "a" ? "Réponse — pédagogique" : "Réponse — technique";
    return (
      <div className="msg-bubble" style={{ width: "100%" }}>
        <div className="answer-card">
          <span className="answer-label">{label}</span>
          <div>{chosenText}</div>
          <SourcesList sources={turn.sources} />
        </div>
        <FeedbackBlock
          question={turn.question}
          answer={chosenText}
          sources={(turn.sources || []).map((s) => s.citation)}
          profile={profile}
        />
      </div>
    );
  }

  return (
    <div className="msg-bubble" style={{ width: "100%" }}>
      <span className="pill-tag">Quelle réponse préférez-vous ?</span>
      <div className="choice-row">
        <div className="answer-card">
          <span className="answer-label">Réponse A — pédagogique</span>
          <div>{turn.answer_a}</div>
          <button className="choose-btn" onClick={() => choose("a")} disabled={saving}>
            Choisir A
          </button>
        </div>
        <div className="answer-card">
          <span className="answer-label">Réponse B — technique</span>
          <div>{turn.answer_b}</div>
          <button className="choose-btn" onClick={() => choose("b")} disabled={saving}>
            Choisir B
          </button>
        </div>
      </div>
      <SourcesList sources={turn.sources} />
    </div>
  );
}

export default function ChatView({ profile, conversationId, onConversationsChange }) {
  const userId = getUserId();
  // Une conversation = une clé locale dédiée. C'est la source de vérité pour
  // "quels tours appartiennent à quelle conversation" — le backend ne stocke
  // qu'un historique plat par utilisateur (voir conversations.js).
  const localKey = `etafakna_conv_${conversationId}`;

  const [turns, setTurns] = useState(() => {
    try {
      const saved = localStorage.getItem(localKey);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);

  // Persiste cette conversation localement, et met à jour son entrée dans
  // l'index (titre + horodatage) affiché dans la sidebar.
  useEffect(() => {
    const clean = turns.filter((t) => !t.pending);
    try {
      localStorage.setItem(localKey, JSON.stringify(clean));
    } catch {
      /* stockage plein/indisponible : sans conséquence pour la session en cours */
    }
    if (clean.length > 0) {
      const updated = upsertConversation(profile, {
        id: conversationId,
        title: titleFromTurns(clean),
        updatedAt: Date.now(),
      });
      onConversationsChange?.(updated);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [turns]);

  const ask = async (question) => {
    const q = question.trim();
    if (!q || loading) return;
    setInput("");
    setError(null);
    setLoading(true);
    const optimisticIndex = turns.length;
    setTurns((t) => [...t, { question: q, pending: true }]);
    try {
      const res = await api.chat(q, profile, userId);
      setTurns((t) => {
        const next = [...t];
        next[optimisticIndex] = { question: q, ...res };
        return next;
      });
    } catch (e) {
      setError(
        e?.response?.data?.detail ||
          "Impossible de contacter le backend. Vérifiez que l'API tourne (voir README)."
      );
      setTurns((t) => t.filter((_, i) => i !== optimisticIndex));
    } finally {
      setLoading(false);
      setTimeout(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
      }, 50);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    ask(input);
  };

  const empty = turns.length === 0;

  return (
    <div className="main-scroll" ref={scrollRef}>
      {empty ? (
        <div className="hero">
          <div className="hero-avatar">⚖️</div>
          <h1>Bonjour</h1>
          <p>
            Votre copilote juridique — posez une question ou explorez les textes de loi.
            <br />
            Profil actif : <strong>{profile.replace("_", " ")}</strong>
          </p>

          <div className="tab-pills">
            <div className="tab-pill active">💬 Chat</div>
            <div className="tab-pill">⚖️ Recherche législative</div>
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              placeholder="Posez votre question…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(input);
                }
              }}
              rows={1}
            />
            <div className="composer-row">
              <div className="composer-left">
                <button type="button" className="icon-btn">
                  <FiPlus />
                </button>
                <button type="button" className="icon-btn">
                  <FiPaperclip />
                </button>
              </div>
              <button type="submit" className={`send-btn ${input.trim() ? "ready" : ""}`}>
                <FiSend />
              </button>
            </div>
            <div className="composer-meta">
              <span>{input.length} / 500</span>
              <span>Nouvelle conversation</span>
            </div>
          </form>

          <div className="disclaimer">L'IA peut se tromper. Les réponses ne constituent pas un conseil juridique.</div>

          {error && <div className="error-banner">{error}</div>}

          <div className="suggestion-grid">
            {SUGGESTIONS.map((s, i) => (
              <button key={i} className="suggestion-card" onClick={() => ask(s.desc)}>
                <div className="suggestion-icon">
                  <s.icon />
                </div>
                <div>
                  <div className="suggestion-title">{s.title}</div>
                  <div className="suggestion-desc">{s.desc}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="conversation">
          {error && <div className="error-banner">{error}</div>}

          {turns.map((turn, i) => (
            <div key={i}>
              <div className="msg-row user">
                <div className="msg-bubble">{turn.question}</div>
              </div>
              <div className="msg-row assistant">
                <div className="assistant-avatar">⚖️</div>
                <div className="assistant-content">
                  {turn.pending ? (
                    <div className="answer-card" style={{ width: "100%" }}>
                      <div className="loading-dots">
                        <span />
                        <span />
                        <span />
                      </div>
                    </div>
                  ) : turn.mode === "single" ? (
                    <SingleAnswer turn={turn} profile={profile} />
                  ) : turn.mode === "smalltalk" ? (
                    <SmallTalkAnswer turn={turn} />
                  ) : (
                    <ChoiceAnswer
                      turn={turn}
                      profile={profile}
                      onChosen={(which) =>
                        setTurns((t) => {
                          const next = [...t];
                          next[i] = { ...next[i], chosen: which };
                          return next;
                        })
                      }
                    />
                  )}
                </div>
              </div>
            </div>
          ))}

          <form className="composer" onSubmit={handleSubmit} style={{ marginTop: 8 }}>
            <textarea
              placeholder="Posez votre question…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(input);
                }
              }}
              rows={1}
            />
            <div className="composer-row">
              <div className="composer-left">
                <button type="button" className="icon-btn">
                  <FiPlus />
                </button>
                <button type="button" className="icon-btn">
                  <FiPaperclip />
                </button>
              </div>
              <button type="submit" className={`send-btn ${input.trim() ? "ready" : ""}`}>
                <FiSend />
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}