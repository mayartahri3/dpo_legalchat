"""
api.py — API FastAPI (mêmes routes et formats de réponse qu'avant).

    ARTICLES_DIR, DB_PATH, EMBED_MODEL, LLM_MODEL, TOP_K, MAX_NEW_TOKENS, EMBED_CACHE_DIR,
    FORCE_BASE_MODEL, N8N_WEBHOOK_URL, N8N_WEBHOOK_TOKEN, AUTO_TRAIN_VIA_N8N,
    PUBLIC_API_URL, EXTERNAL_API_TOKEN, AUTO_TRAIN_MIN_EXAMPLES, AUTO_TRAIN_EPOCHS

Lancement :  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dashboard import compute_metrics
from dataset_builder import build_preference_dataset, training_columns
from dpo_train import DPOTrainingPipeline
from feedback_store import (
    Correction, CorrectionStatus, Feedback, FeedbackStore, Rating, Reason, ResponseStyle,
    StylePreference, UserProfile, can_propose_correction,
)
from grpo_train import GRPOTrainingPipeline
from minirag import EmbeddingIndex, LocalLLM, load_articles
from model_eval import compare_model_versions
from ppo_train import PPOTrainingPipeline
from reward_model import RewardModelPipeline, RewardModelRegistry
from train_common import ModelRegistry, ModelVersion, count_usable_pairs
from validation import ValidationModule

# --------------------------------------------------------------------------- #
# Config — tous les chemins relatifs (models/, reward_models/, datasets/…) sont
# résolus par rapport à ce dossier, quel que soit le répertoire de lancement.
# --------------------------------------------------------------------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def _path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


def _flag(name: str) -> bool:
    return os.environ.get(name, "false").lower() == "true"


ARTICLES_DIR = _path(os.environ.get("ARTICLES_DIR", "articles"))
DB_PATH = _path(os.environ.get("DB_PATH", "feedback.db"))
EMBED_CACHE_DIR = _path(os.environ.get("EMBED_CACHE_DIR", "embedding_cache"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TOP_K = int(os.environ.get("TOP_K", "3"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "500"))
FORCE_BASE_MODEL = _flag("FORCE_BASE_MODEL")

N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "")
N8N_WEBHOOK_TOKEN = os.environ.get("N8N_WEBHOOK_TOKEN", "")
AUTO_TRAIN_VIA_N8N = _flag("AUTO_TRAIN_VIA_N8N")
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", "http://localhost:8000")
EXTERNAL_API_TOKEN = os.environ.get("EXTERNAL_API_TOKEN", "")
AUTO_TRAIN_MIN_EXAMPLES = int(os.environ.get("AUTO_TRAIN_MIN_EXAMPLES", "20"))
AUTO_TRAIN_EPOCHS = int(os.environ.get("AUTO_TRAIN_EPOCHS", "1"))

STYLE_PROMPTS = {
    ResponseStyle.PEDAGOGICAL: (
        "Tu es un assistant juridique. Réponds UNIQUEMENT à partir des articles de loi fournis "
        "en contexte. Utilise un langage simple et pédagogique, avec un exemple concret si utile, "
        "comme si tu expliquais à quelqu'un sans formation juridique. Cite quand même le numéro "
        "d'article utilisé."
    ),
    ResponseStyle.TECHNICAL: (
        "Tu es un assistant juridique. Réponds UNIQUEMENT à partir des articles de loi fournis "
        "en contexte. Utilise un langage technique et précis, structuré, avec référence "
        "systématique et rigoureuse au numéro d'article et à la loi concernée, comme pour un "
        "professionnel du droit."
    ),
}

# --------------------------------------------------------------------------- #
# Small talk — réponse directe, sans RAG ni LLM
# --------------------------------------------------------------------------- #

_TALK_ADVICE = "posez-moi une question sur un texte de loi tunisien et je m'appuierai sur les articles pertinents."
SMALL_TALK = {  # catégorie -> (regex, réponse)
    "greeting": (r"salut+|bonjour|bonsoir|coucou|hello|hi|hey|yo|slt",
                 f"Bonjour ! Je suis votre copilote juridique — {_TALK_ADVICE}"),
    "how_are_you": (r"(?:comment\s*)?(?:ça|ca)\s*va",
                    "Je vais bien, merci ! Et vous ? N'hésitez pas à me poser une question juridique."),
    "thanks": (r"merci+(?:\s*beaucoup)?",
               "Avec plaisir ! N'hésitez pas si vous avez d'autres questions juridiques."),
    "ack": (r"ok(?:ay)?", "D'accord ! Posez-moi votre question quand vous voulez."),
    "goodbye": (r"au revoir|bye|à bientôt|a bientot",
                "Au revoir ! N'hésitez pas à revenir si vous avez d'autres questions juridiques."),
    "identity": (r"qui es[- ]tu|c'?est quoi ce chatbot|tu fais quoi",
                 "Je suis votre copilote juridique — je réponds à vos questions sur les textes de "
                 "loi tunisiens, avec citation systématique des articles."),
    "test": (r"test", "Je fonctionne correctement ! Posez-moi une question juridique quand vous voulez."),
}
_SMALL_TALK_RE = {k: re.compile(rf"^(?:{p})\s*[!.?]*$", re.IGNORECASE) for k, (p, _) in SMALL_TALK.items()}


def small_talk_reply(question: str) -> str | None:
    q = question.strip()
    if not q:
        return SMALL_TALK["greeting"][1]
    if len(q) <= 25:
        for category, rx in _SMALL_TALK_RE.items():
            if rx.match(q):
                return SMALL_TALK[category][1]
    return None


# --------------------------------------------------------------------------- #
# État global + démarrage
# --------------------------------------------------------------------------- #

S: dict = {"auto_last_count": 0, "jobs": {}}
_jobs_lock = threading.Lock()


def _resolve_active_model() -> str:
    if FORCE_BASE_MODEL:
        print("[api] FORCE_BASE_MODEL=true -> modèle de base.")
        return LLM_MODEL
    active = ModelRegistry().active_version()
    if active is None:
        print("[api] Aucune version promue -> modèle de base.")
        return LLM_MODEL
    if not os.path.isdir(active["path"]):
        print(f"[api] ⚠ v{active['version']} introuvable ({active['path']}) -> modèle de base.")
        return LLM_MODEL
    print(f"[api] Version active : v{active['version']} -> {active['path']}")
    return active["path"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    index = EmbeddingIndex(EMBED_MODEL)
    index.build(load_articles(ARTICLES_DIR), cache_dir=EMBED_CACHE_DIR)
    print(f"[api] {len(index.articles)} article(s) indexé(s).")

    model_path = _resolve_active_model()
    S.update(index=index, loaded_model_path=model_path,
             llm=LocalLLM(model_path, max_new_tokens=MAX_NEW_TOKENS, base_model_for_adapter=LLM_MODEL))
    print("[api] Prêt.")
    yield


app = FastAPI(title="E-Tafakna — API RAG juridique", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],   # dev uniquement : restreindre en production
                   allow_methods=["*"], allow_headers=["*"])


def get_store():
    store = FeedbackStore(DB_PATH)
    try:
        yield store
    finally:
        store.close()


def _profile(value: str) -> UserProfile:
    try:
        return UserProfile(value)
    except ValueError:
        raise HTTPException(400, f"Profil inconnu : {value}")


def _check_external_token(token: str | None) -> None:
    if EXTERNAL_API_TOKEN and token != EXTERNAL_API_TOKEN:
        raise HTTPException(401, "Token invalide ou manquant (en-tête X-Import-Token).")


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
    method: str = "dpo"          # dpo | ppo | grpo | reward_model
    fast_test: bool = False


class CompareRequest(BaseModel):
    versions: list[int]
    n_questions: int = 15


class N8nTriggerRequest(BaseModel):
    min_examples: int = 20
    epochs: int = 1


# --------------------------------------------------------------------------- #
# Chat / historique / feedback
# --------------------------------------------------------------------------- #

@app.post("/api/chat")
def chat(req: ChatRequest, store: FeedbackStore = Depends(get_store)):
    profile = _profile(req.profile)
    index, llm = S["index"], S["llm"]
    base = {"question": req.question}

    def save(mode: str, **kw) -> str:
        return store.add_conversation_turn(user_id=req.user_id, profile=profile,
                                           question=req.question, mode=mode, **kw)

    reply = small_talk_reply(req.question)
    if reply is not None:
        return {**base, "mode": "smalltalk", "answer": reply, "sources": [],
                "turn_id": save("smalltalk", answer=reply, sources=[])}

    hits = index.search(req.question, top_k=TOP_K)
    articles = [a for a, _ in hits]
    sources = [{"citation": a.citation(), "score": round(s, 3)} for a, s in hits]

    if can_propose_correction(profile):
        answer = llm.generate(req.question, articles)
        return {**base, "mode": "single", "answer": answer, "sources": sources,
                "turn_id": save("single", answer=answer, sources=sources)}

    # Citoyen / anonyme : deux styles, générés dans UN seul appel batché (§5.5).
    prompts = [llm.build_prompt(req.question, articles, STYLE_PROMPTS[s])
               for s in (ResponseStyle.PEDAGOGICAL, ResponseStyle.TECHNICAL)]
    answer_a, answer_b = llm.generate_prompts(prompts)
    return {**base, "mode": "choice", "answer_a": answer_a, "answer_b": answer_b, "sources": sources,
            "turn_id": save("choice", answer_a=answer_a, answer_b=answer_b, sources=sources)}


@app.get("/api/history")
def history(user_id: str, store: FeedbackStore = Depends(get_store)):
    return {"turns": [{
        "turn_id": r["id"], "profile": r["profile"], "question": r["question"], "mode": r["mode"],
        "answer": r["answer"], "answer_a": r["answer_a"], "answer_b": r["answer_b"],
        "chosen": r["chosen"], "sources": json.loads(r["sources"] or "[]"), "timestamp": r["timestamp"],
    } for r in store.conversation_history(user_id)]}


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest, store: FeedbackStore = Depends(get_store)):
    profile = _profile(req.profile)
    try:
        fb = store.add_feedback(Feedback(
            question=req.question, answer=req.answer,
            rating=Rating(req.rating), user_profile=profile,
            reason=Reason(req.reason) if req.reason else None,
            reason_text=req.reason_text, sources=req.sources,
        ))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"feedback_id": fb.id,
            "can_propose_correction": can_propose_correction(profile) and req.rating == "down"}


@app.post("/api/correction")
def submit_correction(req: CorrectionRequest, store: FeedbackStore = Depends(get_store)):
    try:
        c = store.add_correction(Correction(
            feedback_id=req.feedback_id, question=req.question, rejected_answer=req.rejected_answer,
            proposed_answer=req.proposed_answer, author_profile=_profile(req.profile),
        ))
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return {"correction_id": c.id, "status": c.status.value}


@app.post("/api/style-preference")
def submit_style_preference(req: StylePreferenceRequest, store: FeedbackStore = Depends(get_store)):
    pref = store.add_style_preference(StylePreference(
        question=req.question, answer_a=req.answer_a, style_a=ResponseStyle.PEDAGOGICAL,
        answer_b=req.answer_b, style_b=ResponseStyle.TECHNICAL,
        chosen=req.chosen, user_profile=_profile(req.profile),
    ))
    if req.turn_id:
        store.set_turn_choice(req.turn_id, req.chosen)
    return {"preference_id": pref.id}


# --------------------------------------------------------------------------- #
# Validation experte
# --------------------------------------------------------------------------- #

@app.get("/api/review/pending")
def review_pending(store: FeedbackStore = Depends(get_store)):
    validator = ValidationModule(S["index"], store)
    out = []
    for row in store.pending_corrections():
        c = Correction.from_row(row)
        if c.status == CorrectionStatus.PENDING:
            c = validator.validate(c)
        if c.status == CorrectionStatus.NEEDS_EXPERT:
            out.append({
                "id": c.id, "question": c.question, "rejected_answer": c.rejected_answer,
                "proposed_answer": c.proposed_answer, "author_profile": c.author_profile.value,
                "confidence_score": c.confidence_score, "consensus_count": c.consensus_count,
                "rag_check": c.rag_check,
            })
    return {"pending": out, "auto_train_job_id": _maybe_autotrigger_training(store)}


@app.post("/api/review/decide")
def review_decide(req: ReviewDecisionRequest, store: FeedbackStore = Depends(get_store)):
    correction = store.get_correction(req.correction_id)
    if correction is None:
        raise HTTPException(404, "Correction introuvable.")
    try:
        correction = ValidationModule(S["index"], store).expert_decide(
            correction, req.decision, req.validator_role, req.reformulated_answer)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": correction.id, "status": correction.status.value,
            "auto_train_job_id": _maybe_autotrigger_training(store)}


# --------------------------------------------------------------------------- #
# Dashboard / articles / santé
# --------------------------------------------------------------------------- #

@app.get("/api/dashboard")
def dashboard(store: FeedbackStore = Depends(get_store)):
    m = compute_metrics(store)
    return {
        "total_feedback": m.total_feedback, "satisfaction_rate": m.satisfaction_rate,
        "pending_corrections": m.pending_corrections, "validation_rate": m.validation_rate,
        "reason_breakdown": m.reason_breakdown,
        "style_preference_breakdown": m.style_preference_breakdown, "alerts": m.alerts,
    }


def _article_dict(a, score: float | None = None) -> dict:
    d = {"id": a.id, "title": f"Art. {a.article_number} — {a.title_french}",
         "related_law": f"loi n°{a.law_number}", "chapter": a.chapter,
         "legal_domains": a.legal_domains, "keywords": a.keywords}
    if score is not None:
        d["score"] = round(score, 3)
    return d


@app.get("/api/articles")
def list_articles():
    articles = S["index"].articles
    return {"total": len(articles), "articles": [_article_dict(a) for a in articles]}


@app.get("/api/articles/search")
def search_articles(q: str, top_k: int = 30):
    return {"articles": [_article_dict(a, s) for a, s in S["index"].search(q, top_k=top_k)]}


@app.get("/api/health")
def health():
    return {"status": "ok", "articles_dir": ARTICLES_DIR, "top_k": TOP_K,
            "loaded_model_path": S.get("loaded_model_path", "?")}


@app.delete("/api/seed-demo-data")
def delete_seed_demo_data(store: FeedbackStore = Depends(get_store)):
    return {"deleted": store.delete_seeded_demo_data()}


# --------------------------------------------------------------------------- #
# Entraînement (jobs en arrière-plan)
# --------------------------------------------------------------------------- #

PIPELINES = {"dpo": DPOTrainingPipeline, "ppo": PPOTrainingPipeline, "grpo": GRPOTrainingPipeline}


def _run_training_job(job_id: str, min_examples: int, epochs: int, method: str, fast_test: bool) -> None:
    job = S["jobs"][job_id]
    job["status"] = "running"
    store = FeedbackStore(DB_PATH)
    try:
        n_usable = count_usable_pairs(store)
        if n_usable < min_examples:
            job.update(status="skipped",
                       message=f"Pas assez de corrections validées ({n_usable} < {min_examples}).")
            return
        kwargs = dict(min_examples=min_examples, epochs=epochs, fast_test=fast_test)
        if method == "reward_model":
            rv = RewardModelPipeline(base_model=LLM_MODEL).run_cycle(store, S["index"], **kwargs)
            job.update(status="completed", version=rv.version, path=rv.path,
                       n_training_pairs=rv.n_training_pairs, method=method)
            return
        if method not in PIPELINES:
            raise ValueError(f"Méthode d'entraînement inconnue : {method}")
        mv = PIPELINES[method](base_model=LLM_MODEL).run_cycle(store, S["index"], **kwargs)
        job.update(status="completed", version=mv.version, path=mv.path,
                   n_training_pairs=mv.n_training_pairs, model_status=mv.status, method=mv.method)
    except Exception as e:  # noqa: BLE001 — toute erreur d'entraînement est exposée au frontend
        job.update(status="failed", error=str(e))
    finally:
        store.close()


def _launch_training_job(min_examples: int, epochs: int, trigger: str, method: str = "dpo",
                         fast_test: bool = False) -> str | None:
    with _jobs_lock:   # un seul entraînement à la fois
        if any(j["status"] in ("queued", "running") for j in S["jobs"].values()):
            return None
        job_id = uuid.uuid4().hex[:12]
        S["jobs"][job_id] = {"status": "queued", "min_examples": min_examples, "epochs": epochs,
                             "trigger": trigger, "method": method, "fast_test": fast_test}
    threading.Thread(target=_run_training_job, args=(job_id, min_examples, epochs, method, fast_test),
                     daemon=True).start()
    return job_id


def _notify_n8n(reason: str, min_examples: int, epochs: int) -> dict:
    if not N8N_WEBHOOK_URL:
        raise RuntimeError("N8N_WEBHOOK_URL n'est pas configuré (.env).")
    import requests
    payload = {
        "event": "dpo_train_trigger", "reason": reason, "min_examples": min_examples, "epochs": epochs,
        "base_model": LLM_MODEL,
        "dataset_export_url": f"{PUBLIC_API_URL}/api/dataset/export?min_examples={min_examples}",
        "model_import_url": f"{PUBLIC_API_URL}/api/models/import", "import_token": EXTERNAL_API_TOKEN,
    }
    headers = {"Content-Type": "application/json"}
    if N8N_WEBHOOK_TOKEN:
        headers["X-Webhook-Token"] = N8N_WEBHOOK_TOKEN
    resp = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return {"sent": True, "payload": payload, "n8n_status_code": resp.status_code}


def _maybe_autotrigger_training(store: FeedbackStore) -> Optional[str]:
    n = count_usable_pairs(store)
    if n < AUTO_TRAIN_MIN_EXAMPLES or n == S["auto_last_count"]:
        return None
    S["auto_last_count"] = n
    if AUTO_TRAIN_VIA_N8N:
        try:
            _notify_n8n("auto_threshold_reached", AUTO_TRAIN_MIN_EXAMPLES, AUTO_TRAIN_EPOCHS)
        except Exception as e:  # noqa: BLE001
            print(f"[api] Échec notification n8n (auto) : {e}")
        return None
    return _launch_training_job(AUTO_TRAIN_MIN_EXAMPLES, AUTO_TRAIN_EPOCHS, trigger="auto")


@app.post("/api/train")
def train(req: TrainRequest):
    job_id = _launch_training_job(req.min_examples, req.epochs, "manual", req.method, req.fast_test)
    if job_id is None:
        raise HTTPException(409, "Un entraînement est déjà en cours.")
    return {"job_id": job_id, "status": "queued", "method": req.method}


@app.post("/api/train/trigger-n8n")
def trigger_n8n(req: N8nTriggerRequest):
    try:
        return _notify_n8n("manual", req.min_examples, req.epochs)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Échec de l'appel au webhook n8n : {e}")


@app.get("/api/train/status/{job_id}")
def train_status(job_id: str):
    job = S["jobs"].get(job_id)
    if job is None:
        raise HTTPException(404, "Job introuvable.")
    return job


@app.get("/api/train/jobs")
def train_jobs():
    return {"jobs": [{"job_id": jid, **job} for jid, job in reversed(list(S["jobs"].items()))]}


@app.get("/api/train/stats")
def train_stats(store: FeedbackStore = Depends(get_store)):
    validated = store.validated_corrections()
    return {
        "total_corrections": store.scalar("SELECT COUNT(*) FROM corrections"),
        "demo_corrections": store.scalar(
            "SELECT COUNT(*) FROM corrections WHERE validator_role = 'seed_demo_data'"),
        "validated_corrections": len(validated),
        "validated_corrections_real": sum(1 for r in validated if r["validator_role"] != "seed_demo_data"),
        "usable_training_pairs": count_usable_pairs(store),
        "auto_train_threshold": AUTO_TRAIN_MIN_EXAMPLES,
        "auto_train_via_n8n": AUTO_TRAIN_VIA_N8N,
        "n8n_configured": bool(N8N_WEBHOOK_URL),
    }


# --------------------------------------------------------------------------- #
# Modèles / dataset
# --------------------------------------------------------------------------- #

@app.post("/api/models/compare")
def compare_models(req: CompareRequest, store: FeedbackStore = Depends(get_store)):
    registry = ModelRegistry()
    by_version = {v["version"]: v for v in registry.history()}
    selected = [by_version[v] for v in req.versions if v in by_version]
    if len(selected) < 2:
        raise HTTPException(400, "Sélectionnez au moins deux versions valides à comparer.")
    try:
        result = compare_model_versions(store, S["index"], selected, base_model=LLM_MODEL,
                                        n_questions=req.n_questions)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Échec de la comparaison : {e}")
    for r in result["results"]:
        registry.set_eval_summary(r["version"], {
            "mean_similarity": r["mean_similarity"], "win_rate": r["win_rate"],
            "n_questions": r["n_questions"], "pref_acc": r["pref_acc"], "margin": r["margin"],
            "split": result["split"], "compared_against": req.versions,
        })
    return result


@app.get("/api/models/history")
def models_history():
    return {"history": ModelRegistry().history()}


@app.get("/api/reward-model/history")
def reward_model_history():
    return {"history": RewardModelRegistry().history()}


@app.post("/api/promote/{version}")
def promote(version: int):
    try:
        ModelRegistry().promote(version)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"version": version, "status": "active",
            "note": "Redémarrez l'API (uvicorn) pour charger le modèle promu en mémoire."}


@app.get("/api/dataset/export")
def dataset_export(min_examples: int = 1, x_import_token: str | None = Header(default=None),
                   store: FeedbackStore = Depends(get_store)):
    _check_external_token(x_import_token)
    try:
        examples = build_preference_dataset(store, S["index"], S["llm"].tokenizer,
                                            min_examples=min_examples, top_k=TOP_K)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"n_examples": len(examples), "base_model": LLM_MODEL, "examples": training_columns(examples)}


@app.post("/api/models/import")
async def import_model(
    file: UploadFile = File(...), base_model: str = Form(default=""),
    n_training_pairs: int = Form(default=0), dataset_version: int = Form(default=0),
    method: str = Form(default="dpo"), x_import_token: str | None = Header(default=None),
):
    _check_external_token(x_import_token)
    registry = ModelRegistry()
    version = registry.next_version()
    out_dir = os.path.join(BASE_DIR, "models", f"v{version}")
    os.makedirs(out_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(await file.read())) as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        raise HTTPException(400, "Le fichier envoyé n'est pas une archive .zip valide.")

    registry.register(ModelVersion(
        version=version, base_model=base_model or LLM_MODEL,
        created_at=datetime.now(timezone.utc).isoformat(), dataset_version=dataset_version,
        n_training_pairs=n_training_pairs, path=out_dir, status="staged", method=method,
    ))
    return {"version": version, "path": out_dir, "status": "staged"}