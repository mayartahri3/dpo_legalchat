"""
Legal RAG chatbot — simulation over local JSON legal-article files.

Architecture:
    JSON files  -->  DataLoader  -->  EmbeddingIndex (retrieval)
                                            |
                                            v
                                     RAGChatbot.ask()
                                            |
                                            v
                                LocalLLM (generation, grounded in
                                retrieved articles)

Run:
    pip install -r requirements.txt
    python legal_rag.py --data ./articles
    # then just type your question at the "Question>" prompt and press Enter
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Data model + loading
# --------------------------------------------------------------------------- #

@dataclass
class LegalArticle:
    """Minimal view of one article JSON — only the fields the pipeline needs."""
    id: str
    article_number: str
    title_french: str
    law_number: str
    chapter: str
    content_french: str
    embedding_text: str          # bilingual, summary-enriched — best field to embed
    keywords: list[str] = field(default_factory=list)
    legal_domains: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict) -> "LegalArticle":
        return cls(
            id=data["id"],
            article_number=data.get("article_number", "?"),
            title_french=data.get("title_french", ""),
            law_number=data.get("law_number", ""),
            chapter=data.get("chapter", ""),
            content_french=data.get("content_french", ""),
            embedding_text=data.get("embedding_text") or data.get("content_combined", ""),
            keywords=data.get("keywords", []),
            legal_domains=data.get("legal_domains", []),
        )

    def citation(self) -> str:
        return f"Art. {self.article_number} — {self.title_french} (loi n°{self.law_number})"


def load_articles(data_dir: str | Path) -> list[LegalArticle]:
    """Read every *.json file in data_dir into a LegalArticle."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in {data_dir}")

    articles = []
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        articles.append(LegalArticle.from_json(raw))
    return articles


# --------------------------------------------------------------------------- #
# 2. Embedding index (retrieval)
# --------------------------------------------------------------------------- #

class EmbeddingIndex:
    """Wraps a sentence-transformers model + a FAISS inner-product index.

    Vectors are L2-normalized on encode, so inner product == cosine similarity —
    faiss.IndexFlatIP gives exact nearest-neighbour search (no approximation),
    just backed by FAISS's optimized C++ implementation instead of a numpy
    matmul. At the scale this project targets (dozens to low thousands of
    articles) FlatIP is already fast; swap to IndexIVFFlat/IndexHNSWFlat later
    if the corpus grows into the hundreds of thousands.

    Embeddings are cached to disk (cache_dir) keyed by a hash of the model
    name + every article's id/content, so restarting the server (e.g.
    `uvicorn --reload`) doesn't re-embed the whole corpus every time — only
    the FAISS index is (re)built from the cached vectors, or from scratch on
    first run / whenever the underlying articles change.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.articles: list[LegalArticle] = []
        self.index = None  # faiss.IndexFlatIP, set in build()

    def _cache_key(self, articles: list[LegalArticle]) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(self.model_name.encode("utf-8"))
        for a in articles:
            h.update(a.id.encode("utf-8"))
            h.update(a.embedding_text.encode("utf-8"))
        return h.hexdigest()[:20]

    def _cache_paths(self, cache_dir: str | Path, key: str) -> tuple[Path, Path]:
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.faiss", d / f"{key}.npy"

    def build(self, articles: list[LegalArticle], cache_dir: str | Path | None = "embedding_cache") -> None:
        import faiss

        self.articles = articles
        key = self._cache_key(articles) if cache_dir else None

        if cache_dir:
            index_path, vecs_path = self._cache_paths(cache_dir, key)
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                return

        texts = [a.embedding_text for a in articles]
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.ascontiguousarray(np.asarray(vecs, dtype="float32"))

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self.index = index

        if cache_dir:
            index_path, vecs_path = self._cache_paths(cache_dir, key)
            faiss.write_index(index, str(index_path))
            np.save(vecs_path, vecs)  # gardé pour debug/ré-inspection, pas requis pour la recherche

    def search(self, query: str, top_k: int = 3) -> list[tuple[LegalArticle, float]]:
        if self.index is None:
            raise RuntimeError("Call build() before search().")
        q_vec = self.model.encode([query], normalize_embeddings=True)
        q_vec = np.ascontiguousarray(np.asarray(q_vec, dtype="float32"))
        scores, idxs = self.index.search(q_vec, top_k)
        return [
            (self.articles[i], float(scores[0][pos]))
            for pos, i in enumerate(idxs[0])
            if i != -1
        ]


# --------------------------------------------------------------------------- #
# 3. Generation (local LLM, grounded in retrieved articles)
# --------------------------------------------------------------------------- #

class LocalLLM:
    """Thin wrapper around a local HF text-generation model (e.g. Mistral/Llama/Qwen).

    Swap model_name for whatever you can run locally. Small instruction-tuned
    models (Qwen2.5-1.5B-Instruct, Phi-3-mini) run fine on CPU for a demo;
    Mistral-7B/Llama-3-8B want a GPU (or a quantized GGUF build via llama-cpp-python).
    """

    SYSTEM_PROMPT = (
        "Tu es un assistant juridique. Réponds UNIQUEMENT à partir des articles "
        "de loi fournis en contexte. Si le contexte ne contient pas la réponse, "
        "dis-le clairement. Cite systématiquement le numéro d'article utilisé."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 500,
                 base_model_for_adapter: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from pathlib import Path

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        # dpo_train.py entraîne avec use_lora=True -> le dossier sauvegardé
        # (models/vN/) ne contient qu'un ADAPTATEUR LoRA (adapter_config.json
        # + poids d'adaptateur), pas un modèle complet. AutoModelForCausalLM
        # .from_pretrained() plante dessus. On détecte ce cas et on fusionne
        # l'adaptateur sur le modèle de base avant de servir l'inférence.
        is_lora_adapter = (Path(model_name) / "adapter_config.json").exists()

        if is_lora_adapter:
            from peft import PeftModel
            import json
            base_name = base_model_for_adapter
            if base_name is None:
                # Le nom du modèle de base est stocké dans adapter_config.json
                # (champ base_model_name_or_path) — pas besoin de le redonner
                # explicitement si on l'a bien à l'entraînement.
                cfg = json.loads((Path(model_name) / "adapter_config.json").read_text())
                base_name = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # tokenizer sauvegardé avec l'adaptateur
            base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32)
            peft_model = PeftModel.from_pretrained(base_model, model_name)
            self.model = peft_model.merge_and_unload()  # fusionne LoRA -> poids denses, inférence normale ensuite
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            )
        self.model.eval()  # inference mode: disables dropout etc.

    def generate(self, query: str, context_articles: list[LegalArticle],
                 system_prompt: str | None = None) -> str:
        context = "\n\n".join(
            f"[{a.citation()}]\n{a.content_french}" for a in context_articles
        )
        messages = [
            {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {query}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # no_grad: skip building the autograd graph, not needed for inference
        # do_sample=False (greedy): skips sampling overhead, deterministic output
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

def build_chat_prompt(tokenizer, query: str, context_articles: list, system_prompt: str) -> str:
    """
    Build the exact same prompt format that LocalLLM.generate() uses.
    
    This ensures DPO/PPO/GRPO training uses the same prompt distribution
    as inference, maintaining consistency between training and production.
    
    Args:
        tokenizer: The tokenizer from the LLM
        query: User question
        context_articles: List of retrieved LegalArticle objects
        system_prompt: System prompt to use
    
    Returns:
        Formatted prompt string ready for tokenization
    """
    context = "\n\n".join(
        f"[{a.citation()}]\n{a.content_french}" for a in context_articles
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {query}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
# --------------------------------------------------------------------------- #
# 4. Orchestrator
# --------------------------------------------------------------------------- #

class RAGChatbot:
    def __init__(self, index: EmbeddingIndex, llm: LocalLLM, top_k: int = 3):
        self.index = index
        self.llm = llm
        self.top_k = top_k

    def ask(self, query: str) -> dict:
        hits = self.index.search(query, top_k=self.top_k)
        articles = [a for a, _score in hits]
        answer = self.llm.generate(query, articles)
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {"citation": a.citation(), "score": round(score, 3)}
                for a, score in hits
            ],
            "articles": articles,  # utilisé par legal_rag_app.py pour régénérer d'autres styles
        }


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Legal RAG chatbot (simulation)")
    parser.add_argument("--data", default="articles", help="Directory containing the 25 article .json files (default: ./articles)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    print(f"Loading articles from {args.data} ...")
    articles = load_articles(args.data)
    print(f"Loaded {len(articles)} articles.")

    print("Building embedding index ...")
    index = EmbeddingIndex(args.embed_model)
    index.build(articles)

    print("Loading local LLM (first run downloads weights) ...")
    llm = LocalLLM(args.llm_model)

    bot = RAGChatbot(index, llm, top_k=args.top_k)

    def show(result: dict):
        print("\n--- Réponse ---")
        print(result["answer"])
        print("\n--- Sources ---")
        for s in result["sources"]:
            print(f"  ({s['score']}) {s['citation']}")
        print()

    print("Prêt. Tapez votre question et appuyez sur Entrée ('quit' pour sortir).\n")
    while True:
        q = input("Question> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            show(bot.ask(q))


if __name__ == "__main__":
    main()