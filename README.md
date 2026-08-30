# E-Tafakna — Interface web (React) + backend RAG existant

Ce dossier ajoute une **interface web** au-dessus de ton pipeline existant.
**Aucun fichier `.py` fourni n'a été modifié** : `minirag.py`, `feedback_store.py`,
`validation.py`, `dataset_builder.py`, `dpo_train.py`, `dashboard.py` et
`legal_rag_app.py` sont copiés tels quels dans `backend/`. Le seul ajout est
`backend/api.py`, une fine couche FastAPI qui appelle exactement les mêmes
fonctions/classes que `legal_rag_app.py` (mêmes imports, même logique
`can_propose_correction`, mêmes `STYLE_SYSTEM_PROMPTS`, même `ValidationModule`,
etc.), simplement exposées en HTTP au lieu de boucles `input()`.

```
etafakna/
├── backend/
│   ├── minirag.py            (inchangé)
│   ├── feedback_store.py     (inchangé)
│   ├── validation.py         (inchangé)
│   ├── dataset_builder.py    (inchangé)
│   ├── dpo_train.py          (inchangé)
│   ├── dashboard.py          (inchangé)
│   ├── legal_rag_app.py      (inchangé, CLI toujours fonctionnel en parallèle)
│   ├── api.py                (NOUVEAU — wrapper HTTP)
│   ├── requirements.txt       (+ fastapi/uvicorn)
│   └── articles/             (⚠️ à remplacer — voir ci-dessous)
└── frontend/                  (React + Vite, style repris de la maquette)
```

## ⚠️ Étape obligatoire : brancher ton vrai dossier `legaldata`

`backend/articles/` ne contient que **l'unique fichier JSON d'exemple** que tu
as partagé (`00a56862-...json`), pour que le projet démarre. Il n'a pas été
question de recréer tes articles — remplace ce dossier par ton vrai dossier
`legaldata` complet avant de lancer l'API :

```bash
rm -rf backend/articles
cp -r /chemin/vers/ton/legaldata backend/articles
# ou : export ARTICLES_DIR=/chemin/vers/ton/legaldata (voir variables d'env plus bas)
```

`minirag.load_articles()` lit tous les `*.json` de ce dossier exactement comme
avant — aucun changement de format attendu, tes fichiers ont déjà la bonne
structure (`embedding_text`, `content_french`, `article_number`, etc.).

## Lancer le backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt



uvicorn api:app --reload --port 8000
```

Le premier démarrage télécharge les poids du modèle d'embedding et du LLM
(accès réseau Hugging Face requis, comme pour `python legal_rag_app.py chat`
avant). Une fois démarré : `http://localhost:8000/api/health`.

Le CLI d'origine reste utilisable en parallèle si besoin :
`python legal_rag_app.py --data ./articles chat`.

## Lancer le frontend

```bash
cd frontend
npm install
cp .env.example .env     # VITE_API_URL=http://localhost:8000
npm run dev
```

Ouvre `http://localhost:5173`.

## Ce que couvre l'interface

| Écran | Route API | Section cahier des charges |
|---|---|---|
| Sélecteur de profil (sidebar) | — | §4 remplace `ask_profile()` du CLI |
| Chat — avocat_verifie/juriste_interne : une réponse + correction | `POST /api/chat`, `/api/feedback`, `/api/correction` | §5.1, §5.2 |
| Chat — citoyen/anonyme : choix A/B pédagogique vs technique | `POST /api/chat`, `/api/style-preference`, `/api/feedback` | §5.5 |
| Validation experte (file d'attente) | `GET /api/review/pending`, `POST /api/review/decide` | §5.3, §4 |
| Tableau de bord | `GET /api/dashboard` | §5.6 |

`POST /api/train` et `/api/promote/{version}` exposent aussi le cycle DPO
(`dpo_train.py`) si tu veux les déclencher depuis un outil externe (ex.
Postman/curl) — pas encore reliés à un écran dédié dans cette première
version de l'UI, volontairement, pour rester focalisé sur la boucle
question → réponse → feedback → correction → validation.

## Vérifications faites avant livraison

- `backend/api.py` s'importe sans dépendre de `sentence-transformers`/`torch`
  au chargement (comme `minirag.py` d'origine : les imports lourds sont dans
  les constructeurs `EmbeddingIndex`/`LocalLLM`, appelés seulement au
  démarrage réel du serveur).
- Toutes les routes ont été testées avec un index/LLM factices (même
  principe que ton `test_pipeline.py`) : `/chat` (single + choice),
  `/feedback`, `/correction` (y compris le 403 pour un profil `citoyen`),
  `/style-preference`, `/review/pending`, `/dashboard`.
- `npm run build` du frontend passe sans erreur.

**Note indépendante (non corrigée, hors périmètre de cette tâche)** : ton
`test_pipeline.py` fourni attend un `ValueError` autour de
`store.add_correction(...)` pour un profil `citoyen`, mais
`feedback_store.py` lève un `PermissionError` — le test plantera tel quel.
`api.py` gère bien `PermissionError` (→ HTTP 403), donc l'API n'est pas
affectée ; à corriger dans `test_pipeline.py` si tu veux que ce fichier
passe de nouveau.
