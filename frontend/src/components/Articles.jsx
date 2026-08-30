import { useEffect, useMemo, useState } from "react";
import { FiSearch } from "react-icons/fi";
import api from "../api";

function useDebounced(value, delay = 350) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

function ArticleCard({ a }) {
  return (
    <div className="review-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div style={{ fontWeight: 600, fontSize: 14.5 }}>{a.title}</div>
        {a.score !== undefined && (
          <span className="answer-label" style={{ marginBottom: 0, flexShrink: 0 }}>
            pertinence {a.score}
          </span>
        )}
      </div>
      <div style={{ marginTop: 6 }}>
        <span className="pill-tag">{a.related_law}</span>
      </div>
      {a.chapter && (
        <div className="fb-note" style={{ marginTop: 8 }}>
          {a.chapter}
        </div>
      )}
      {a.legal_domains?.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
          {a.legal_domains.map((d, i) => (
            <span key={i} className="reason-chip" style={{ cursor: "default" }}>
              {d}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Articles() {
  const [all, setAll] = useState([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null); // null = pas de recherche active -> affiche `all`
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState(null);

  const debouncedQuery = useDebounced(query, 350);

  useEffect(() => {
    api
      .listArticles()
      .then((res) => {
        setAll(res.articles);
        setTotal(res.total);
      })
      .catch(() => setError("Impossible de charger les articles."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const q = debouncedQuery.trim();
    if (!q) {
      setResults(null);
      return;
    }
    setSearching(true);
    api
      .searchArticles(q, 40)
      .then((res) => setResults(res.articles))
      .catch(() => setError("Erreur lors de la recherche."))
      .finally(() => setSearching(false));
  }, [debouncedQuery]);

  // Filtre client instantané en plus de la recherche sémantique (debounced) —
  // donne un retour immédiat pendant que la requête serveur part en tâche de fond.
  const instantFiltered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return all;
    return all.filter(
      (a) =>
        a.title.toLowerCase().includes(q) ||
        a.related_law.toLowerCase().includes(q) ||
        a.chapter?.toLowerCase().includes(q) ||
        a.keywords?.some((k) => k.toLowerCase().includes(q))
    );
  }, [all, query]);

  const shown = results ?? instantFiltered;

  return (
    <div className="main-scroll">
      <div className="page">
        <h2>Articles</h2>
        <p className="page-sub">
          {total} article{total > 1 ? "s" : ""} chargé{total > 1 ? "s" : ""} depuis le dossier
          legaldata. La recherche combine un filtre instantané (titre, loi, mots-clés) et une
          recherche par pertinence sémantique sur l'index utilisé par le chat.
        </p>

        <div className="composer" style={{ padding: "10px 14px", marginBottom: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <FiSearch style={{ color: "var(--text-faint)", flexShrink: 0 }} />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Rechercher un article, une loi, un mot-clé…"
              style={{ border: "none", outline: "none", fontSize: 14.5, width: "100%" }}
            />
            {searching && <span className="fb-note">recherche…</span>}
          </div>
        </div>

        {error && <div className="error-banner">{error}</div>}

        {loading ? (
          <div className="empty-state">Chargement des articles…</div>
        ) : shown.length === 0 ? (
          <div className="empty-state">Aucun article ne correspond à cette recherche.</div>
        ) : (
          shown.map((a) => <ArticleCard key={a.id} a={a} />)
        )}
      </div>
    </div>
  );
}
