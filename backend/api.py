"""
api.py — Interface HTTP (FastAPI) au-dessus du pipeline existant.

IMPORTANT : ce fichier ne modifie AUCUN module existant. Il réutilise tel
quel minirag.py (EmbeddingIndex, LocalLLM, RAGChatbot, load_articles),
feedback_store.py (FeedbackStore, Feedback, Correction, StylePreference,
can_propose_correction), validation.py (ValidationModule), dashboard.py
(compute_metrics) et dataset_builder/dpo_train pour /train. C'est la même
logique que legal_rag_app.py (chat / review / train / dashboard), simplement
appelée depuis des routes HTTP au lieu de boucles input().

Config via variables d'environnement (mêmes défauts que legal_rag_app.py) :
    ARTICLES_DIR   dossier des articles JSON (§ legaldata)   default: "articles"
    DB_PATH        fichier sqlite                            default: "feedback.db"
    EMBED_MODEL    modèle sentence-transformers               default: all-MiniLM-L6-v2
    LLM_MODEL      modèle HF causal-LM local                  default: Qwen/Qwen2.5-1.5B-Instruct
    TOP_K          nb d'articles récupérés par question       default: 3

Lancement :
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv()  # lit un fichier .env dans le dossier courant s'il existe

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feedback_store import (
    Correction, CorrectionStatus, Feedback, FeedbackStore, Rating, Reason,
    ResponseStyle, StylePreference, UserProfile, can_propose_correction,
)
from validation import ValidationModule
from dashboard import compute_metrics
from dpo_train import DPOTrainingPipeline, ModelRegistry
from reward_model import RewardModelPipeline, RewardModelRegistry
from ppo_train import PPOTrainingPipeline
from grpo_train import GRPOTrainingPipeline
from model_eval import compare_model_versions

from minirag import LocalLLM, RAGChatbot, EmbeddingIndex, load_articles

# Réutilise les mêmes system-prompts de style que legal_rag_app.py (§5.5)
from legal_rag_app import STYLE_SYSTEM_PROMPTS

import re

# --------------------------------------------------------------------------- #
# Détection de messages de "small talk" (salutations, remerciements, etc.) —
# évite de lancer la recherche RAG + les deux générations A/B (§5.5) pour des
# messages qui ne posent pas de question juridique. Répond directement, sans
# appeler le LLM, donc quasi instantané, et sans demander de préférence de
# style puisqu'il n'y a pas deux réponses à comparer.
#
# NOUVEAU : une seule réponse générique pour TOUT type de small talk donnait
# des réponses absurdes ("Bonjour !" en réponse à "merci beaucoup") — chaque
# catégorie (salutation / remerciement / au revoir / ça va / identité / test)
# a maintenant sa propre réponse adaptée.
# --------------------------------------------------------------------------- #

SMALL_TALK_CATEGORIES = [
    ("greeting", [
        r"^salut+\s*[!.?]*$", r"^bonjour\s*[!.?]*$", r"^bonsoir\s*[!.?]*$",
        r"^coucou\s*[!.?]*$", r"^(hello|hi|hey|yo)\s*[!.?]*$", r"^slt\s*[!.?]*$",
    ]),
    ("how_are_you", [
        r"^(ça|ca)\s*va\s*[?!.]*$", r"^comment\s*(ça|ca)\s*va\s*[?!.]*$",
    ]),
    ("thanks", [
        r"^merci+\s*[!.]*$", r"^merci\s*beaucoup\s*[!.]*$",
    ]),
    ("ack", [
        r"^ok(ay)?\s*[!.]*$",
    ]),
    ("goodbye", [
        r"^(au revoir|bye|à bientôt|a bientot)\s*[!.]*$",
    ]),
    ("identity", [
        r"^(qui es[- ]tu|c'?est quoi ce chatbot|tu fais quoi)\s*[?]*$",
    ]),
    ("test", [
        r"^test\s*[!.?]*$",
    ]),
]

_SMALL_TALK_RE = [
    (category, re.compile(p, re.IGNORECASE))
    for category, patterns in SMALL_TALK_CATEGORIES
    for p in patterns
]

SMALL_TALK_REPLIES = {
    "greeting": (
        "Bonjour ! Je suis votre copilote juridique — posez-moi une question sur un texte de "
        "loi tunisien (droit du travail, propriété, entreprise, etc.) et je vous répondrai en "
        "m'appuyant sur les articles pertinents."
    ),
    "how_are_you": (
        "Je vais bien, merci ! Et vous ? N'hésitez pas à me poser une question juridique dès "
        "que vous êtes prêt(e)."
    ),
    "thanks": (
        "Avec plaisir ! N'hésitez pas si vous avez d'autres questions juridiques."
    ),
    "ack": "D'accord ! Posez-moi votre question quand vous voulez.",
    "goodbye": "Au revoir ! N'hésitez pas à revenir si vous avez d'autres questions juridiques.",
    "identity": (
        "Je suis votre copilote juridique — je réponds à vos questions sur les textes de loi "
        "tunisiens en m'appuyant sur les articles pertinents, avec citation systématique."
    ),
    "test": "Je fonctionne correctement ! Posez-moi une question juridique quand vous voulez.",
}
# Repli si jamais une catégorie n'a pas de réponse dédiée (ne devrait pas arriver).
DEFAULT_SMALL_TALK_REPLY = SMALL_TALK_REPLIES["greeting"]


def small_talk_category(question: str) -> str | None:
    """Renvoie la catégorie de small talk détectée (ou None si la question doit
    passer par le pipeline RAG normal)."""
    q = question.strip()
    if not q:
        return "greeting"  # message vide -> traité comme une salutation d'accueil
    if len(q) <= 25:
        for category, pattern in _SMALL_TALK_RE:
            if pattern.match(q):
                return category
    return None


def is_small_talk(question: str) -> bool:
    """Conservé pour compatibilité — préférer small_talk_category()."""
    return small_talk_category(question) is not None


# --------------------------------------------------------------------------- #
# Config (identique aux défauts argparse de legal_rag_app.py)
# --------------------------------------------------------------------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    """Résout un chemin relatif par rapport au dossier de api.py (BASE_DIR),
    pas par rapport au cwd du process qui lance uvicorn."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


ARTICLES_DIR = _resolve(os.environ.get("ARTICLES_DIR", "articles"))
DB_PATH = _resolve(os.environ.get("DB_PATH", "feedback.db"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TOP_K = int(os.environ.get("TOP_K", "3"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "500"))
EMBED_CACHE_DIR = _resolve(os.environ.get("EMBED_CACHE_DIR", "embedding_cache"))

# Si True, on ignore la version 'active' de ModelRegistry et on force
# toujours le modèle de base LLM_MODEL — utile pour comparer base vs.
# fine-tuné, ou pour revenir en arrière rapidement sans dé-promouvoir.
FORCE_BASE_MODEL = os.environ.get("FORCE_BASE_MODEL", "false").lower() == "true"

# --------------------------------------------------------------------------- #
# n8n
# --------------------------------------------------------------------------- #

N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "")
N8N_WEBHOOK_TOKEN = os.environ.get("N8N_WEBHOOK_TOKEN", "")
AUTO_TRAIN_VIA_N8N = os.environ.get("AUTO_TRAIN_VIA_N8N", "false").lower() == "true"
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", "http://localhost:8000")
EXTERNAL_API_TOKEN = os.environ.get("EXTERNAL_API_TOKEN", "")


def _check_external_token(x_import_token: str | None) -> None:
    if EXTERNAL_API_TOKEN and x_import_token != EXTERNAL_API_TOKEN:
        raise HTTPException(401, "Token invalide ou manquant (en-tête X-Import-Token).")

app = FastAPI(title="E-Tafakna — API RAG juridique")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev local uniquement ; restreindre en production
    allow_methods=["*"],
    allow_headers=["*"],
)

_gen_executor = ThreadPoolExecutor(max_workers=2)

# --------------------------------------------------------------------------- #
# État global — construit une seule fois au démarrage (comme cmd_chat)
# --------------------------------------------------------------------------- #

_state: dict = {}


def _resolve_active_model() -> str:
    """
    NOUVEAU : au lieu de toujours charger LLM_MODEL (modèle de base), on
    vérifie d'abord si une version a été promue ('active') dans
    ModelRegistry. Si oui, c'est ELLE qu'on charge — sinon on retombe sur
    LLM_MODEL. C'est ce qui manquait pour que "Promouvoir en production"
    (Training.jsx / legal_rag_app.py promote) ait un effet réel sur /api/chat.
    """
    if FORCE_BASE_MODEL:
        print("[api] FORCE_BASE_MODEL=true -> modèle de base forcé, version promue ignorée.")
        return LLM_MODEL

    active = ModelRegistry().active_version()
    if active is None:
        print("[api] Aucune version promue trouvée dans ModelRegistry -> modèle de base utilisé.")
        return LLM_MODEL

    active_path = active["path"]
    if not os.path.isdir(active_path):
        print(f"[api] ATTENTION : version active v{active['version']} pointe vers "
              f"'{active_path}' introuvable sur ce disque -> modèle de base utilisé à la place.")
        return LLM_MODEL

    print(f"[api] Version active trouvée : v{active['version']} -> {active_path}")
    return active_path


@app.on_event("startup")
def startup() -> None:
    print(f"[api] DB_PATH          = {DB_PATH}")
    print(f"[api] ARTICLES_DIR     = {ARTICLES_DIR}")
    print(f"[api] EMBED_CACHE_DIR  = {EMBED_CACHE_DIR}")

    print(f"[api] Chargement des articles depuis '{ARTICLES_DIR}' ...")
    articles = load_articles(ARTICLES_DIR)
    print(f"[api] {len(articles)} article(s) chargé(s).")

    index = EmbeddingIndex(EMBED_MODEL)
    index.build(articles, cache_dir=EMBED_CACHE_DIR)

    model_path = _resolve_active_model()
    # base_model_for_adapter : si model_path est un adaptateur LoRA, LocalLLM
    # lit déjà le modèle de base dans adapter_config.json — on le passe quand
    # même en secours explicite au cas où ce champ serait absent.
    llm = LocalLLM(model_path, max_new_tokens=MAX_NEW_TOKENS, base_model_for_adapter=LLM_MODEL)
    bot = RAGChatbot(index, llm, top_k=TOP_K)

    _state["index"] = index
    _state["llm"] = llm
    _state["bot"] = bot
    _state["loaded_model_path"] = model_path

    store = get_store()
    n_feedback = store.conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
    n_corrections = store.conn.execute("SELECT COUNT(*) AS n FROM corrections").fetchone()["n"]
    n_validated = len(store.validated_corrections())
    store.close()
    print(f"[api] Base existante   : {n_feedback} feedback(s), {n_corrections} correction(s), "
          f"{n_validated} validée(s).")
    print("[api] Prêt.")


def get_store() -> FeedbackStore:
    return FeedbackStore(DB_PATH)


def _profile(value: str) -> UserProfile:
    try:
        return UserProfile(value)
    except ValueError:
        raise HTTPException(400, f"Profil inconnu : {value}")


# --------------------------------------------------------------------------- #
# Schémas
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    question: str
    profile: str
    user_id: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    profile: str
    reason: Optional[str] = None
    reason_text: Optional[str] = None
    sources: list[str] = []


class CorrectionRequest(BaseModel):
    feedback_id: str
    question: str
    rejected_answer: str
    proposed_answer: str
    profile: str


class StylePreferenceRequest(BaseModel):
    question: str
    answer_a: str
    answer_b: str
    chosen: Literal["a", "b"]
    profile: str
    turn_id: Optional[str] = None


class ReviewDecisionRequest(BaseModel):
    correction_id: str
    decision: Literal["valide", "rejete", "reformule"]
    validator_role: str = "expert_juriste"
    reformulated_answer: Optional[str] = None


class TrainRequest(BaseModel):
    min_examples: int = 20
    epochs: int = 1
    method: str = "dpo"  # "dpo" | "ppo" | "grpo" | "reward_model"
    fast_test: bool = False  # cap à quelques steps/séquences courtes — pour valider la mécanique du pipeline rapidement


class SeedDemoDataRequest(BaseModel):
    n: int = 6


# --------------------------------------------------------------------------- #
# §5.1/§5.2/§5.5 — /chat
# --------------------------------------------------------------------------- #

@app.post("/api/chat")
def chat(req: ChatRequest):
    profile = _profile(req.profile)
    bot: RAGChatbot = _state["bot"]
    index: EmbeddingIndex = _state["index"]
    llm: LocalLLM = _state["llm"]
    store = get_store()

    if can_propose_correction(profile):
        result = bot.ask(req.question)
        turn_id = store.add_conversation_turn(
            user_id=req.user_id, profile=profile, question=req.question, mode="single",
            answer=result["answer"], sources=result["sources"],
        )
        store.close()
        return {
            "mode": "single",
            "question": req.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "turn_id": turn_id,
        }

    category = small_talk_category(req.question)
    if category is not None:
        reply = SMALL_TALK_REPLIES.get(category, DEFAULT_SMALL_TALK_REPLY)
        turn_id = store.add_conversation_turn(
            user_id=req.user_id, profile=profile, question=req.question, mode="smalltalk",
            answer=reply, sources=[],
        )
        store.close()
        return {
            "mode": "smalltalk",
            "question": req.question,
            "answer": reply,
            "sources": [],
            "turn_id": turn_id,
        }

    hits = index.search(req.question, top_k=TOP_K)
    hit_articles = [a for a, _score in hits]
    future_a = _gen_executor.submit(
        llm.generate, req.question, hit_articles, system_prompt=STYLE_SYSTEM_PROMPTS[ResponseStyle.PEDAGOGICAL]
    )
    future_b = _gen_executor.submit(
        llm.generate, req.question, hit_articles, system_prompt=STYLE_SYSTEM_PROMPTS[ResponseStyle.TECHNICAL]
    )
    answer_a = future_a.result()
    answer_b = future_b.result()
    sources = [{"citation": a.citation(), "score": round(score, 3)} for a, score in hits]

    turn_id = store.add_conversation_turn(
        user_id=req.user_id, profile=profile, question=req.question, mode="choice",
        answer_a=answer_a, answer_b=answer_b, sources=sources,
    )
    store.close()
    return {
        "mode": "choice",
        "question": req.question,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "sources": sources,
        "turn_id": turn_id,
    }


@app.get("/api/history")
def history(user_id: str):
    store = get_store()
    rows = store.conversation_history(user_id)
    store.close()
    import json
    out = []
    for r in rows:
        out.append({
            "turn_id": r["id"],
            "profile": r["profile"],
            "question": r["question"],
            "mode": r["mode"],
            "answer": r["answer"],
            "answer_a": r["answer_a"],
            "answer_b": r["answer_b"],
            "chosen": r["chosen"],
            "sources": json.loads(r["sources"] or "[]"),
            "timestamp": r["timestamp"],
        })
    return {"turns": out}


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    profile = _profile(req.profile)
    store = get_store()
    try:
        fb = store.add_feedback(Feedback(
            question=req.question,
            answer=req.answer,
            rating=Rating.UP if req.rating == "up" else Rating.DOWN,
            user_profile=profile,
            reason=Reason(req.reason) if req.reason else None,
            reason_text=req.reason_text,
            sources=req.sources,
        ))
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        store.close()

    return {
        "feedback_id": fb.id,
        "can_propose_correction": can_propose_correction(profile) and req.rating == "down",
    }


@app.post("/api/correction")
def submit_correction(req: CorrectionRequest):
    profile = _profile(req.profile)
    store = get_store()
    try:
        correction = store.add_correction(Correction(
            feedback_id=req.feedback_id,
            question=req.question,
            rejected_answer=req.rejected_answer,
            proposed_answer=req.proposed_answer,
            author_profile=profile,
        ))
    except PermissionError as e:
        raise HTTPException(403, str(e))
    finally:
        store.close()
    return {"correction_id": correction.id, "status": correction.status.value}


@app.post("/api/style-preference")
def submit_style_preference(req: StylePreferenceRequest):
    profile = _profile(req.profile)
    store = get_store()
    try:
        pref = store.add_style_preference(StylePreference(
            question=req.question,
            answer_a=req.answer_a, style_a=ResponseStyle.PEDAGOGICAL,
            answer_b=req.answer_b, style_b=ResponseStyle.TECHNICAL,
            chosen=req.chosen, user_profile=profile,
        ))
        if req.turn_id:
            store.set_turn_choice(req.turn_id, req.chosen)
    finally:
        store.close()
    return {"preference_id": pref.id}


@app.get("/api/review/pending")
def review_pending():
    store = get_store()
    validator = ValidationModule(_state["index"], store)
    rows = store.pending_corrections()

    out = []
    for row in rows:
        correction = Correction(
            id=row["id"], feedback_id=row["feedback_id"], question=row["question"],
            rejected_answer=row["rejected_answer"], proposed_answer=row["proposed_answer"],
            author_profile=UserProfile(row["author_profile"]),
            status=CorrectionStatus(row["status"]), timestamp=row["timestamp"],
        )
        if correction.status == CorrectionStatus.PENDING:
            correction = validator.validate(correction)
        if correction.status == CorrectionStatus.NEEDS_EXPERT:
            out.append({
                "id": correction.id,
                "question": correction.question,
                "rejected_answer": correction.rejected_answer,
                "proposed_answer": correction.proposed_answer,
                "author_profile": correction.author_profile.value,
                "confidence_score": correction.confidence_score,
                "consensus_count": correction.consensus_count,
                "rag_check": correction.rag_check,
            })
    auto_job_id = _maybe_autotrigger_training(store)
    store.close()
    return {"pending": out, "auto_train_job_id": auto_job_id}


@app.post("/api/review/decide")
def review_decide(req: ReviewDecisionRequest):
    store = get_store()
    validator = ValidationModule(_state["index"], store)
    cur = store.conn.execute("SELECT * FROM corrections WHERE id = ?", (req.correction_id,)).fetchone()
    if cur is None:
        store.close()
        raise HTTPException(404, "Correction introuvable.")

    correction = Correction(
        id=cur["id"], feedback_id=cur["feedback_id"], question=cur["question"],
        rejected_answer=cur["rejected_answer"], proposed_answer=cur["proposed_answer"],
        author_profile=UserProfile(cur["author_profile"]),
        status=CorrectionStatus(cur["status"]), timestamp=cur["timestamp"],
        confidence_score=cur["confidence_score"], rag_check=cur["rag_check"],
        consensus_count=cur["consensus_count"],
    )
    try:
        correction = validator.expert_decide(
            correction, req.decision, req.validator_role, req.reformulated_answer,
        )
    except ValueError as e:
        store.close()
        raise HTTPException(400, str(e))
    auto_job_id = _maybe_autotrigger_training(store)
    store.close()
    return {"id": correction.id, "status": correction.status.value, "auto_train_job_id": auto_job_id}


@app.get("/api/dashboard")
def dashboard():
    store = get_store()
    m = compute_metrics(store)
    store.close()
    return {
        "total_feedback": m.total_feedback,
        "satisfaction_rate": m.satisfaction_rate,
        "pending_corrections": m.pending_corrections,
        "validation_rate": m.validation_rate,
        "reason_breakdown": m.reason_breakdown,
        "style_preference_breakdown": m.style_preference_breakdown,
        "alerts": m.alerts,
    }


@app.delete("/api/seed-demo-data")
def delete_seed_demo_data():
    store = get_store()
    deleted = store.delete_seeded_demo_data()
    store.close()
    return {"deleted": deleted}


AUTO_TRAIN_MIN_EXAMPLES = int(os.environ.get("AUTO_TRAIN_MIN_EXAMPLES", "20"))
AUTO_TRAIN_EPOCHS = int(os.environ.get("AUTO_TRAIN_EPOCHS", "1"))

_train_jobs: dict[str, dict] = {}
_auto_train_last_count = 0


def _run_training_job(job_id: str, min_examples: int, epochs: int, method: str = "dpo",
                       fast_test: bool = False) -> None:
    """
    method: "dpo" | "ppo" | "grpo" | "reward_model" — les quatre partagent le
    même mécanisme de job en arrière-plan (thread + polling par le frontend),
    seule la pipeline appelée change. C'est ce qui permet de lancer/suivre
    les trois méthodes de comparaison depuis la même page Entraînement.

    fast_test=True : cap à quelques steps / séquences courtes sur chaque
    pipeline — sert à vérifier que le job démarre, tourne et se termine
    correctement (mécanique du pipeline) sans attendre un vrai cycle complet,
    surtout utile pour PPO/GRPO qui sont nettement plus lents que DPO.
    """
    _train_jobs[job_id]["status"] = "running"
    store = get_store()
    index: EmbeddingIndex = _state["index"]
    try:
        if method == "reward_model":
            pipeline = RewardModelPipeline(base_model=LLM_MODEL)
            rv = pipeline.run_cycle(store, index, min_examples=min_examples, epochs=epochs, fast_test=fast_test)
            _train_jobs[job_id].update(status="completed", version=rv.version, path=rv.path,
                                        n_training_pairs=rv.n_training_pairs, method="reward_model")
            return

        if method == "dpo":
            pipeline = DPOTrainingPipeline(base_model=LLM_MODEL)
            if not pipeline.should_trigger(store, min_examples=min_examples):
                _train_jobs[job_id].update(status="skipped", message=f"Pas assez de corrections validées (< {min_examples}).")
                return
            mv = pipeline.run_cycle(store, index, min_examples=min_examples, epochs=epochs, fast_test=fast_test)
        elif method == "ppo":
            pipeline = PPOTrainingPipeline(base_model=LLM_MODEL)
            mv = pipeline.run_cycle(store, index, min_examples=min_examples, epochs=epochs, fast_test=fast_test)
        elif method == "grpo":
            pipeline = GRPOTrainingPipeline(base_model=LLM_MODEL)
            mv = pipeline.run_cycle(store, index, min_examples=min_examples, epochs=epochs, fast_test=fast_test)
        else:
            raise ValueError(f"Méthode d'entraînement inconnue : {method}")

        _train_jobs[job_id].update(
            status="completed", version=mv.version, path=mv.path,
            n_training_pairs=mv.n_training_pairs, model_status=mv.status, method=mv.method,
        )
    except Exception as e:  # noqa: BLE001 — on veut capturer et exposer toute erreur d'entraînement
        _train_jobs[job_id].update(status="failed", error=str(e))
    finally:
        store.close()


def _launch_training_job(min_examples: int, epochs: int, trigger: str, method: str = "dpo",
                          fast_test: bool = False) -> str:
    job_id = uuid.uuid4().hex[:12]
    _train_jobs[job_id] = {
        "status": "queued", "min_examples": min_examples, "epochs": epochs,
        "trigger": trigger, "method": method, "fast_test": fast_test,
    }
    thread = threading.Thread(
        target=_run_training_job, args=(job_id, min_examples, epochs, method, fast_test), daemon=True,
    )
    thread.start()
    return job_id


def _notify_n8n(reason: str, min_examples: int, epochs: int) -> dict:
    if not N8N_WEBHOOK_URL:
        raise RuntimeError(
            "N8N_WEBHOOK_URL n'est pas configuré (.env) — ajoute l'URL du nœud Webhook "
            "de ton workflow n8n pour activer ce déclenchement."
        )
    import requests

    payload = {
        "event": "dpo_train_trigger",
        "reason": reason,
        "min_examples": min_examples,
        "epochs": epochs,
        "base_model": LLM_MODEL,
        "dataset_export_url": f"{PUBLIC_API_URL}/api/dataset/export?min_examples={min_examples}",
        "model_import_url": f"{PUBLIC_API_URL}/api/models/import",
        "import_token": EXTERNAL_API_TOKEN,
    }
    headers = {"Content-Type": "application/json"}
    if N8N_WEBHOOK_TOKEN:
        headers["X-Webhook-Token"] = N8N_WEBHOOK_TOKEN

    resp = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return {"sent": True, "payload": payload, "n8n_status_code": resp.status_code}


def _maybe_autotrigger_training(store: FeedbackStore) -> Optional[str]:
    global _auto_train_last_count
    n_validated = len(store.validated_corrections())
    if n_validated < AUTO_TRAIN_MIN_EXAMPLES:
        return None
    if n_validated == _auto_train_last_count:
        return None
    if any(j["status"] in ("queued", "running") for j in _train_jobs.values()):
        return None
    _auto_train_last_count = n_validated

    if AUTO_TRAIN_VIA_N8N:
        try:
            _notify_n8n("auto_threshold_reached", AUTO_TRAIN_MIN_EXAMPLES, AUTO_TRAIN_EPOCHS)
        except Exception as e:  # noqa: BLE001
            print(f"[api] Échec notification n8n (auto) : {e}")
        return None
    return _launch_training_job(AUTO_TRAIN_MIN_EXAMPLES, AUTO_TRAIN_EPOCHS, trigger="auto")


@app.post("/api/train")
def train(req: TrainRequest):
    """method: "dpo" (défaut, comportement inchangé) | "ppo" | "grpo" | "reward_model" —
    voir TrainRequest.method. Un cycle PPO nécessite qu'un reward model existe déjà
    (POST /api/train avec method="reward_model" d'abord)."""
    job_id = _launch_training_job(
        req.min_examples, req.epochs, trigger="manual", method=req.method, fast_test=req.fast_test
    )
    return {"job_id": job_id, "status": "queued", "method": req.method}


class CompareRequest(BaseModel):
    versions: list[int]           # numéros de version à comparer (voir /api/models/history)
    n_questions: int = 15


@app.post("/api/models/compare")
def compare_models(req: CompareRequest):
    """Compare plusieurs versions (DPO / PPO / GRPO / base) sur les mêmes questions
    validées — similarité d'embedding à la réponse validée + win-rate par paires.
    Coûteux : charge chaque modèle successivement, prévoir plusieurs minutes."""
    registry = ModelRegistry()
    all_versions = {v["version"]: v for v in registry.history()}
    selected = [all_versions[v] for v in req.versions if v in all_versions]
    if len(selected) < 2:
        raise HTTPException(400, "Sélectionnez au moins deux versions valides à comparer.")

    store = get_store()
    try:
        result = compare_model_versions(
            store, _state["index"], selected, base_model=LLM_MODEL, n_questions=req.n_questions,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Échec de la comparaison : {e}")
    finally:
        store.close()

    for r in result["results"]:
        registry.set_eval_summary(r["version"], {
            "mean_similarity": r["mean_similarity"], "win_rate": r["win_rate"],
            "n_questions": r["n_questions"], "compared_against": req.versions,
        })
    return result


@app.get("/api/reward-model/history")
def reward_model_history():
    return {"history": RewardModelRegistry().history()}


class N8nTriggerRequest(BaseModel):
    min_examples: int = 20
    epochs: int = 1


@app.post("/api/train/trigger-n8n")
def trigger_n8n(req: N8nTriggerRequest):
    try:
        result = _notify_n8n("manual", req.min_examples, req.epochs)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Échec de l'appel au webhook n8n : {e}")
    return result


@app.get("/api/train/status/{job_id}")
def train_status(job_id: str):
    job = _train_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job introuvable.")
    return job


@app.get("/api/train/jobs")
def train_jobs():
    return {"jobs": [{"job_id": jid, **job} for jid, job in reversed(list(_train_jobs.items()))]}


@app.get("/api/train/stats")
def train_stats():
    store = get_store()
    total = store.conn.execute("SELECT COUNT(*) AS n FROM corrections").fetchone()["n"]
    demo = store.conn.execute(
        "SELECT COUNT(*) AS n FROM corrections WHERE validator_role = 'seed_demo_data'"
    ).fetchone()["n"]
    validated_rows = store.validated_corrections()
    n_validated = len(validated_rows)
    n_validated_real = sum(1 for r in validated_rows if r["validator_role"] != "seed_demo_data")
    store.close()
    return {
        "total_corrections": total,
        "demo_corrections": demo,
        "validated_corrections": n_validated,
        "validated_corrections_real": n_validated_real,
        "auto_train_threshold": AUTO_TRAIN_MIN_EXAMPLES,
        "auto_train_via_n8n": AUTO_TRAIN_VIA_N8N,
        "n8n_configured": bool(N8N_WEBHOOK_URL),
    }


@app.get("/api/dataset/export")
def dataset_export(min_examples: int = 1, x_import_token: str | None = Header(default=None)):
    _check_external_token(x_import_token)
    from dataset_builder import build_preference_dataset, training_columns
    store = get_store()
    try:
        examples = build_preference_dataset(store, min_examples=min_examples)
    except ValueError as e:
        store.close()
        raise HTTPException(400, str(e))
    store.close()
    return {
        "n_examples": len(examples),
        "base_model": LLM_MODEL,
        "examples": training_columns(examples),
    }


@app.post("/api/models/import")
async def import_model(
    file: UploadFile = File(...),
    base_model: str = Form(default=""),
    n_training_pairs: int = Form(default=0),
    dataset_version: int = Form(default=0),
    method: str = Form(default="dpo"),
    x_import_token: str | None = Header(default=None),
):
    _check_external_token(x_import_token)
    import zipfile
    import io
    from datetime import datetime, timezone
    from dpo_train import ModelVersion

    registry = ModelRegistry()
    version = registry.next_version()
    out_dir = os.path.join(BASE_DIR, "models", f"v{version}")
    os.makedirs(out_dir, exist_ok=True)

    content = await file.read()
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        raise HTTPException(400, "Le fichier envoyé n'est pas une archive .zip valide.")

    mv = ModelVersion(
        version=version,
        base_model=base_model or LLM_MODEL,
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_version=dataset_version,
        n_training_pairs=n_training_pairs,
        path=out_dir,
        status="staged",
        method=method,
    )
    registry.register(mv)
    return {"version": version, "path": out_dir, "status": "staged"}


@app.get("/api/models/history")
def models_history():
    return {"history": ModelRegistry().history()}


@app.post("/api/promote/{version}")
def promote(version: int):
    ModelRegistry().promote(version)
    return {
        "version": version,
        "status": "active",
        "note": "Redémarrez le serveur API (uvicorn) pour que le modèle promu soit "
                "effectivement chargé en mémoire — la promotion met à jour registry.json "
                "mais ne recharge pas le modèle déjà en cours d'exécution.",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "articles_dir": ARTICLES_DIR,
        "top_k": TOP_K,
        "loaded_model_path": _state.get("loaded_model_path", "?"),
    }