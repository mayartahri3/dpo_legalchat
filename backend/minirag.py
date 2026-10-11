"""minirag.py — retrieval (FAISS) + génération locale ancrée dans les articles."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SYSTEM_PROMPT = (
    "Tu es un assistant juridique. Réponds UNIQUEMENT à partir des articles "
    "de loi fournis en contexte. Si le contexte ne contient pas la réponse, "
    "dis-le clairement. Cite systématiquement le numéro d'article utilisé."
)


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #

@dataclass
class LegalArticle:
    id: str
    article_number: str
    title_french: str
    law_number: str
    chapter: str
    content_french: str
    embedding_text: str
    keywords: list[str] = field(default_factory=list)
    legal_domains: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict) -> "LegalArticle":
        return cls(
            id=d["id"],
            article_number=d.get("article_number", "?"),
            title_french=d.get("title_french", ""),
            law_number=d.get("law_number", ""),
            chapter=d.get("chapter", ""),
            content_french=d.get("content_french", ""),
            embedding_text=d.get("embedding_text") or d.get("content_combined", ""),
            keywords=d.get("keywords", []),
            legal_domains=d.get("legal_domains", []),
        )

    def citation(self) -> str:
        return f"Art. {self.article_number} — {self.title_french} (loi n°{self.law_number})"


def load_articles(data_dir: str | Path) -> list[LegalArticle]:
    files = sorted(Path(data_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Aucun .json trouvé dans {data_dir}")
    return [LegalArticle.from_json(json.loads(p.read_text(encoding="utf-8"))) for p in files]


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

class EmbeddingIndex:
    """sentence-transformers + FAISS IndexFlatIP (vecteurs normalisés => cosinus exact).
    L'index est mis en cache sur disque (clé = modèle + contenu des articles)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.articles: list[LegalArticle] = []
        self.index = None

    def build(self, articles: list[LegalArticle], cache_dir: str | Path | None = "embedding_cache") -> None:
        import faiss
        self.articles = articles

        cache_file = None
        if cache_dir:
            h = hashlib.sha256(self.model_name.encode())
            for a in articles:
                h.update(a.id.encode())
                h.update(a.embedding_text.encode())
            cache_file = Path(cache_dir) / f"{h.hexdigest()[:20]}.faiss"
            if cache_file.exists():
                self.index = faiss.read_index(str(cache_file))
                return

        vecs = self.model.encode([a.embedding_text for a in articles],
                                 normalize_embeddings=True, show_progress_bar=False)
        vecs = np.ascontiguousarray(vecs, dtype="float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(cache_file))

    def search(self, query: str, top_k: int = 3) -> list[tuple[LegalArticle, float]]:
        if self.index is None:
            raise RuntimeError("Appelez build() avant search().")
        q = np.ascontiguousarray(self.model.encode([query], normalize_embeddings=True), dtype="float32")
        scores, idxs = self.index.search(q, min(top_k, len(self.articles)))
        return [(self.articles[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]


# --------------------------------------------------------------------------- #
# Prompt + génération
# --------------------------------------------------------------------------- #

def build_chat_prompt(tokenizer, query: str, context_articles: list, system_prompt: str) -> str:
    """Prompt EXACT utilisé à l'inférence ET à l'entraînement (DPO/PPO/GRPO)."""
    context = "\n\n".join(f"[{a.citation()}]\n{a.content_french}" for a in context_articles)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {query}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class LocalLLM:
    """Modèle HF local (modèle complet ou adaptateur LoRA, fusionné au chargement).
    Génération greedy, batchable, protégée par un verrou (le modèle est partagé
    entre les requêtes FastAPI)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 500,
                 base_model_for_adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()

        dtype = torch.float32
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        adapter_cfg = Path(model_name) / "adapter_config.json"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if adapter_cfg.exists():
            from peft import PeftModel
            base_name = base_model_for_adapter or json.loads(adapter_cfg.read_text()).get(
                "base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
            base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, low_cpu_mem_usage=True)
            self.model = PeftModel.from_pretrained(base, model_name).merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # requis pour la génération batchée
        self.model.to(self.device).eval()

    def build_prompt(self, query: str, articles: list[LegalArticle], system_prompt: str | None = None) -> str:
        return build_chat_prompt(self.tokenizer, query, articles, system_prompt or SYSTEM_PROMPT)

    def generate_prompts(self, prompts: list[str]) -> list[str]:
        """Génère une réponse par prompt, en UN seul appel batché."""
        tok = self.tokenizer
        inputs = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with self._lock, self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=tok.pad_token_id)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True)]

    def generate(self, query: str, context_articles: list[LegalArticle],
                 system_prompt: str | None = None) -> str:
        return self.generate_prompts([self.build_prompt(query, context_articles, system_prompt)])[0]


class RAGChatbot:
    def __init__(self, index: EmbeddingIndex, llm: LocalLLM, top_k: int = 3):
        self.index, self.llm, self.top_k = index, llm, top_k

    def ask(self, query: str) -> dict:
        hits = self.index.search(query, top_k=self.top_k)
        answer = self.llm.generate(query, [a for a, _ in hits])
        return {"query": query, "answer": answer,
                "sources": [{"citation": a.citation(), "score": round(s, 3)} for a, s in hits]}