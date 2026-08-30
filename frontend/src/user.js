const USER_ID_KEY = "etafakna_user_id";

function uuid() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  // Fallback pour les environnements sans crypto.randomUUID
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Identité stable par navigateur — pas un vrai compte utilisateur (pas d'auth),
 * mais suffisant pour que l'historique de conversation reste cohérent pour un
 * même utilisateur tant qu'il garde le même navigateur/profil. Une vraie
 * authentification multi-appareils serait une étape suivante.
 */
export function getUserId() {
  let id = localStorage.getItem(USER_ID_KEY);
  if (!id) {
    id = uuid();
    localStorage.setItem(USER_ID_KEY, id);
  }
  return id;
}
