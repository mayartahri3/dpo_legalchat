import {
  FiHome,
  FiMessageSquare,
  FiBarChart2,
  FiCheckSquare,
  FiDownload,
  FiCpu,
  FiBookOpen,
  FiPlus,
  FiClock,
} from "react-icons/fi";
import { PROFILES } from "../constants";

const NAV_ITEMS = [
  { key: "chat", label: "Chat Elyssa", icon: FiMessageSquare },
  { key: "articles", label: "Articles", icon: FiBookOpen },
  { key: "review", label: "Validation experte", icon: FiCheckSquare },
  { key: "dashboard", label: "Tableau de bord", icon: FiBarChart2 },
  { key: "training", label: "Entraînement DPO", icon: FiCpu },
];

export default function Sidebar({
  page,
  onNavigate,
  profile,
  onProfileChange,
  conversations = [],
  activeConversationId,
  onNewConversation,
  onSelectConversation,
}) {
  const currentProfile = PROFILES.find((p) => p.value === profile);
  const initials = currentProfile ? currentProfile.label.slice(0, 1) : "?";

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="sidebar-logo-mark">E</span>
        E-Tafakna
      </div>

      <button className="new-conversation-btn" onClick={onNewConversation}>
        <FiPlus /> Nouvelle conversation
      </button>

      <nav className="sidebar-nav">
        <button className="sidebar-link" disabled title="Non implémenté dans cette démo">
          <FiHome /> Accueil
        </button>
        {NAV_ITEMS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            className={`sidebar-link ${page === key ? "active" : ""}`}
            onClick={() => onNavigate(key)}
          >
            <Icon /> {label}
          </button>
        ))}

        {page === "chat" && (
          <div className="conversation-history">
            <div className="conversation-history-title">
              <FiClock /> Historique
            </div>
            {conversations.length === 0 ? (
              <div className="conversation-empty">Aucune conversation pour l'instant.</div>
            ) : (
              conversations.map((c) => (
                <button
                  key={c.id}
                  className={`conversation-item ${c.id === activeConversationId ? "active" : ""}`}
                  onClick={() => onSelectConversation(c.id)}
                  title={c.title}
                >
                  {c.title}
                </button>
              ))
            )}
          </div>
        )}
      </nav>

      <div className="sidebar-footer">
        <button className="get-app-btn" disabled title="Non implémenté dans cette démo">
          <FiDownload /> Get App
        </button>
        <div className="profile-card">
          <div className="profile-avatar">{initials}</div>
          <div>
            <div className="profile-name">{currentProfile?.label ?? "Profil"}</div>
            <div className="profile-sub">Essai gratuit</div>
          </div>
        </div>
        <select
          className="profile-select"
          value={profile}
          onChange={(e) => onProfileChange(e.target.value)}
        >
          {PROFILES.map((p) => (
            <option key={p.value} value={p.value}>
              {p.label}
            </option>
          ))}
        </select>
      </div>
    </aside>
  );
}