"""
eval_metrics.py — Évaluation chiffrée du base model vs versions DPO/PPO/GRPO,
conçue pour les PETITS datasets (où la similarité d'embedding seule ne
distingue rien).

Métriques (sur le split choisi, par défaut le holdout `test`) :

  Teacher-forced (pas de génération, très sensibles même avec ~5 exemples) :
    - pref_acc : % de paires où la version augmente la vraisemblance de "chosen"
                 PLUS que celle de "rejected" par rapport au modèle de base.
    - margin   : moyenne de (Δlogp_chosen − Δlogp_rejected) vs base.
                 > 0 = la version a appris la préférence. ≈ 0 = n'a rien appris.
    - d_chosen : Δ logp(chosen) vs base.

  Génération (greedy, même prompt RAG qu'à l'entraînement) :
    - emb_sim / rouge_l : proximité à la réponse validée (sémantique / lexicale)
    - cite_rate   : % de réponses qui citent au moins un article
    - halluc_rate : % de réponses qui citent un article ABSENT du contexte RAG
    - cite_recall : part des articles cités par la réponse validée que la version cite aussi
    - distinct3   : diversité (détecte les répétitions / boucles)
    - win_rate    : part des questions où la version a la meilleure emb_sim

  Retrieval (indépendant du LLM) : hit@k et MRR, avec pour vérité terrain les
  articles cités par la réponse validée.

  Intervalles de confiance à 95 % par bootstrap — avec n petit, un écart dont
  l'IC contient 0 n'est PAS une vraie différence.

Usage :
    python eval_metrics.py --make-split --test-frac 0.25     # une fois, AVANT d'entraîner
    python eval_metrics.py --split test                       # généralisation
    python eval_metrics.py --split train                      # mémorisation
    python eval_metrics.py --split test --versions 1 2 3 --write-registry
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np

from dataset_builder import build_preference_dataset, make_holdout_split
from feedback_store import FeedbackStore
from minirag import EmbeddingIndex, LocalLLM, load_articles
from train_common import ModelRegistry

ART_RE = re.compile(r"(?i)\bart(?:icle)?s?\.?\s*(\d+)")


# --------------------------------------------------------------------------- #
# Petits utilitaires de métriques
# --------------------------------------------------------------------------- #

def cited_numbers(text: str) -> set[str]:
    return set(ART_RE.findall(text or ""))


def article_numbers(article) -> set[str]:
    return set(re.findall(r"\d+", str(article.article_number)))


def rouge_l_f1(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y, 1):
            cur.append(prev[j - 1] + 1 if xi == yj else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(x), lcs / len(y)
    return 2 * p * r / (p + r)


def distinct_3(text: str) -> float:
    toks = text.lower().split()
    tris = list(zip(toks, toks[1:], toks[2:]))
    return len(set(tris)) / len(tris) if tris else 1.0


def bootstrap_ci(x, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------- #
# Log-vraisemblance et génération (teacher-forced / greedy)
# --------------------------------------------------------------------------- #

def completion_logprob(llm, prompt: str, completion: str, max_len: int = 768) -> float:
    """Somme des log-probs des tokens de `completion` sachant `prompt`."""
    torch = llm.torch
    tok = llm.tokenizer
    device = next(llm.model.parameters()).device

    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    if len(c_ids) > max_len - 64:
        c_ids = c_ids[: max_len - 64]
    p_ids = p_ids[-(max_len - len(c_ids)):]  # tronque le début du prompt si trop long

    ids = torch.tensor([p_ids + c_ids], device=device)
    with torch.no_grad():
        logits = llm.model(ids).logits[0]
    logprobs = torch.log_softmax(logits[:-1].float(), dim=-1)
    tok_lp = logprobs.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return float(tok_lp[len(p_ids) - 1:].sum())


def generate_from_prompt(llm, prompt: str, max_new_tokens: int) -> str:
    tok = llm.tokenizer
    device = next(llm.model.parameters()).device
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with llm.torch.no_grad():
        out = llm.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Évaluation d'UN modèle
# --------------------------------------------------------------------------- #

def evaluate_model(path: str, base_model: str, examples: list[dict], ctx_nums: list[set[str]],
                   embed_model, max_new_tokens: int) -> dict[str, np.ndarray]:
    import torch

    llm = LocalLLM(path, max_new_tokens=max_new_tokens, base_model_for_adapter=base_model)
    if torch.cuda.is_available():
        llm.model.to("cuda")
    llm.model.eval()

    m: dict[str, list[float]] = {k: [] for k in (
        "lp_chosen", "lp_rejected", "emb_sim", "rouge_l", "cite_any",
        "halluc", "cite_recall", "distinct3", "n_words", "empty",
    )}

    for ex, ctx in zip(examples, ctx_nums):
        m["lp_chosen"].append(completion_logprob(llm, ex["prompt"], ex["chosen"]))
        m["lp_rejected"].append(completion_logprob(llm, ex["prompt"], ex["rejected"]))

        gen = generate_from_prompt(llm, ex["prompt"], max_new_tokens)
        v = embed_model.encode([gen or " ", ex["chosen"]], normalize_embeddings=True)
        m["emb_sim"].append(float(np.dot(v[0], v[1])))
        m["rouge_l"].append(rouge_l_f1(gen, ex["chosen"]))

        cites = cited_numbers(gen)
        gold = cited_numbers(ex["chosen"])
        m["cite_any"].append(float(bool(cites)))
        m["halluc"].append(float(bool(cites - ctx)))
        m["cite_recall"].append(len(cites & gold) / len(gold) if gold else float("nan"))
        m["distinct3"].append(distinct_3(gen))
        m["n_words"].append(len(gen.split()))
        m["empty"].append(float(not gen))

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {k: np.asarray(v, dtype=float) for k, v in m.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="feedback.db")
    ap.add_argument("--data", default="articles")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--versions", type=int, nargs="*", default=None,
                    help="Versions du registry à évaluer (défaut : toutes).")
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--n-questions", type=int, default=0, help="0 = tous les exemples du split.")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--write-registry", action="store_true",
                    help="Écrit mean_similarity/win_rate dans registry.json (affichés par Training.jsx).")
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument("--make-split", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    store = FeedbackStore(args.db)

    if args.make_split:
        split = make_holdout_split(store, test_frac=args.test_frac, seed=args.seed)
        print(f"eval_split.json écrit : {len(split['test_questions'])} question(s) de test "
              f"sur {split['n_questions_total']}.")
        print("⚠ Copiez ce fichier sur Colab (export_for_colab.py) AVANT tout entraînement, "
              "et ré-entraînez les versions : celles déjà entraînées ont vu ces questions.")
        store.close()
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    index = EmbeddingIndex(args.embed_model)
    index.build(load_articles(args.data))

    examples = build_preference_dataset(store, index, tokenizer, min_examples=1,
                                        top_k=args.top_k, split=args.split)
    if args.n_questions:
        examples = examples[: args.n_questions]
    store.close()
    n = len(examples)
    print(f"\nSplit '{args.split}' : {n} paire(s).")
    if n < 8:
        print("⚠ n < 8 : les intervalles de confiance seront très larges — "
              "lisez les écarts comme des indications, pas des preuves.")

    # --- Retrieval (ne dépend d'aucun LLM) ---
    ctx_nums: list[set[str]] = []
    hits_at_k, rr = [], []
    for ex in examples:
        hits = [a for a, _ in index.search(ex["question"], top_k=args.top_k)]
        ctx: set[str] = set()
        for a in hits:
            ctx |= article_numbers(a)
        ctx_nums.append(ctx)
        gold = cited_numbers(ex["chosen"])
        if gold:
            hits_at_k.append(float(bool(gold & ctx)))
            rank = next((i for i, a in enumerate(hits, 1) if gold & article_numbers(a)), None)
            rr.append(1.0 / rank if rank else 0.0)
    if hits_at_k:
        print(f"Retrieval : hit@{args.top_k} = {np.mean(hits_at_k):.0%}   MRR = {np.mean(rr):.3f} "
              f"(sur {len(hits_at_k)} question(s) dont la réponse validée cite un article)")

    # --- Modèles à évaluer : base d'abord (référence), puis les versions ---
    registry = ModelRegistry(args.models_root)
    selected = [v for v in registry.history()
                if args.versions is None or v["version"] in args.versions]
    candidates = [("base", args.base_model, None)] + [
        (f"v{v['version']}-{v.get('method', '?')}", v["path"], v) for v in selected
    ]

    results: dict[str, dict[str, np.ndarray]] = {}
    for label, path, _entry in candidates:
        print(f"\n>>> Évaluation de {label}  ({path})")
        results[label] = evaluate_model(path, args.base_model, examples, ctx_nums,
                                        index.model, args.max_new_tokens)

    # --- Dérivés : marge de récompense implicite vs base ---
    ref = results["base"]
    for label, r in results.items():
        r["d_chosen"] = r["lp_chosen"] - ref["lp_chosen"]
        r["d_rejected"] = r["lp_rejected"] - ref["lp_rejected"]
        r["margin"] = r["d_chosen"] - r["d_rejected"]
        r["pref_acc"] = (args.beta * r["margin"] > 0).astype(float)

    labels = list(results)
    sims = np.stack([results[l]["emb_sim"] for l in labels])
    winners = sims.argmax(axis=0)
    for i, l in enumerate(labels):
        results[l]["win"] = (winners == i).astype(float)

    # --- Affichage ---
    def f(x, pct=False, nd=3):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return "-"
        mean, lo, hi = bootstrap_ci(x)
        return f"{mean:.0%} [{lo:.0%},{hi:.0%}]" if pct else f"{mean:+.{nd}f} [{lo:+.{nd}f},{hi:+.{nd}f}]"

    print("\n" + "=" * 118)
    print(f"RÉSULTATS — split='{args.split}', n={n}  (moyenne [IC95% bootstrap])")
    print("=" * 118)
    print(f"{'version':<14}{'pref_acc':<20}{'margin':<26}{'emb_sim':<24}{'rouge_l':<24}")
    for l in labels:
        r = results[l]
        if l == "base":
            print(f"{l:<14}{'-':<20}{'-':<26}{f(r['emb_sim']):<24}{f(r['rouge_l']):<24}")
        else:
            print(f"{l:<14}{f(r['pref_acc'], pct=True):<20}{f(r['margin'], nd=2):<26}"
                  f"{f(r['emb_sim']):<24}{f(r['rouge_l']):<24}")
    print("-" * 118)
    print(f"{'version':<14}{'cite_rate':<10}{'halluc':<9}{'cite_recall':<13}{'distinct3':<11}"
          f"{'mots':<7}{'vide':<7}{'win_rate':<9}")
    for l in labels:
        r = results[l]
        cr = r["cite_recall"][~np.isnan(r["cite_recall"])]
        print(f"{l:<14}{r['cite_any'].mean():<10.0%}{r['halluc'].mean():<9.0%}"
              f"{(cr.mean() if len(cr) else float('nan')):<13.0%}{r['distinct3'].mean():<11.2f}"
              f"{r['n_words'].mean():<7.0f}{r['empty'].mean():<7.0%}{r['win'].mean():<9.0%}")
    print("=" * 118)
    print("Lecture : pref_acc > 50 % et margin > 0 (IC sans 0) = la version a appris la préférence. "
          "margin ≈ 0 = entraînement trop faible (lr / epochs / nb de steps).\n"
          "halluc et cite_rate ne doivent pas se dégrader vs base ; vide/distinct3 bas = dégénérescence.")

    # --- Sauvegarde ---
    report = {
        "split": args.split, "n": n,
        "retrieval": {"hit_at_k": float(np.mean(hits_at_k)) if hits_at_k else None,
                      "mrr": float(np.mean(rr)) if rr else None},
        "per_version": {l: {k: v.tolist() for k, v in r.items()} for l, r in results.items()},
        "questions": [e["question"] for e in examples],
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport détaillé (par question) : {args.out}")

    if args.write_registry:
        for (label, _p, entry) in candidates:
            if entry is None:
                continue
            r = results[label]
            registry.set_eval_summary(entry["version"], {
                "mean_similarity": float(r["emb_sim"].mean()),
                "win_rate": float(r["win"].mean()),
                "n_questions": n,
                "pref_acc": float(r["pref_acc"].mean()),
                "margin": float(r["margin"].mean()),
                "split": args.split,
            })
        print("Résumés écrits dans registry.json.")


if __name__ == "__main__":
    main()