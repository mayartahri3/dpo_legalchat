// Valeurs alignées exactement sur feedback_store.py — ne pas diverger du backend.

export const PROFILES = [
  { value: "citoyen", label: "Citoyen" },
  { value: "avocat_verifie", label: "Avocat vérifié" },
  { value: "juriste_interne", label: "Juriste interne" },
  { value: "anonyme", label: "Anonyme" },
];

export const CORRECTION_ALLOWED_PROFILES = new Set(["avocat_verifie", "juriste_interne"]);

export const canProposeCorrection = (profile) => CORRECTION_ALLOWED_PROFILES.has(profile);

export const REASONS = [
  { value: "trop_generale", label: "Trop générale" },
  { value: "juridiquement_incorrecte", label: "Juridiquement incorrecte" },
  { value: "incomplete", label: "Incomplète" },
  { value: "hors_sujet", label: "Hors sujet" },
  { value: "autre", label: "Autre" },
];
