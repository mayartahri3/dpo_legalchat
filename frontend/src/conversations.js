// conversations.js — regroupe les échanges par conversation, côté frontend.
//
// Le backend (feedback_store / conversation_turns) garde un historique PLAT
// par user_id, sans notion de conversation distincte — /api/history renvoie
// tous les tours mélangés. Plutôt que de modifier le schéma backend, on gère
// le regroupement en conversations ici, dans localStorage, par profil (un
// citoyen et un avocat sur le même navigateur ont des historiques séparés).
//
// Chaque conversation est {id, title, updatedAt} dans l'index, et son
// contenu (turns) est stocké séparément sous etafakna_conv_{id} — voir
// ChatView.jsx. Ça évite de dupliquer tout le contenu des messages dans
// l'index utilisé par la sidebar.

const INDEX_PREFIX = "etafakna_conversations_";

function indexKey(profile) {
  return `${INDEX_PREFIX}${profile}`;
}

export function loadConversations(profile) {
  try {
    const raw = localStorage.getItem(indexKey(profile));
    const list = raw ? JSON.parse(raw) : [];
    return list.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  } catch {
    return [];
  }
}

function saveConversations(profile, list) {
  try {
    localStorage.setItem(indexKey(profile), JSON.stringify(list));
  } catch {
    /* stockage plein/indisponible : sans conséquence pour la session en cours */
  }
}

/** Crée ou met à jour l'entrée d'index pour une conversation, renvoie la liste à jour. */
export function upsertConversation(profile, { id, title, updatedAt }) {
  const list = loadConversations(profile);
  const idx = list.findIndex((c) => c.id === id);
  const entry = { id, title, updatedAt };
  if (idx === -1) list.unshift(entry);
  else list[idx] = entry;
  saveConversations(profile, list);
  return loadConversations(profile);
}

export function deleteConversation(profile, id) {
  const list = loadConversations(profile).filter((c) => c.id !== id);
  saveConversations(profile, list);
  try {
    localStorage.removeItem(`etafakna_conv_${id}`);
  } catch {
    /* ignore */
  }
  return list;
}

export function newConversationId() {
  return `conv_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

/** Dérive un titre lisible depuis la première question de la conversation. */
export function titleFromTurns(turns) {
  const first = turns.find((t) => t.question)?.question?.trim();
  if (!first) return "Nouvelle conversation";
  return first.length > 46 ? `${first.slice(0, 46)}…` : first;
}
