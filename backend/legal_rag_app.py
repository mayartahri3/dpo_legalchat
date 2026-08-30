"""
legal_rag_app.py — Point d'entrée unique implémentant le cycle du §7 :

    1. question -> 2. réponse (RAG+LLM) -> 3. feedback (+correction OU style) ->
    4. validation -> 5. dataset -> 6. (reward implicite) -> 7. DPO -> 8. déploiement

Sous-commandes :
    chat      boucle de questions/réponses avec collecte de feedback (§5.1) +
              correction de contenu réservée aux avocats/juristes (§5.2) OU
              choix de structure de réponse pour les citoyens/anonymes (§5.5)
    review    file d'attente experte pour les corrections ambiguës (§5.3, §4)
    train     déclenche un cycle DPO si assez de corrections validées (§5.4, §7)
    dashboard affiche les métriques de suivi (§5.6)

Droits par profil (§4, §5.2, §5.5) :
    - avocat_verifie / juriste_interne : peuvent proposer une CORRECTION de
      contenu juridique (passe par le module de validation, §5.3).
    - citoyen / anonyme : n'ont pas la compétence juridique pour corriger le
      fond. Ils choisissent seulement la STRUCTURE de réponse qu'ils
      préfèrent, entre deux variantes générées (pédagogique / technique) —
      façon "quelle réponse préférez-vous" de ChatGPT. Ce choix nourrit un
      dataset de personnalisation séparé (§5.5), jamais le dataset de
      correction juridique.

Ce fichier réutilise minirag.py (load_articles, EmbeddingIndex, LocalLLM,
RAGChatbot) dans le même dossier que les modules ci-dessous.
"""

from __future__ import annotations

import argparse
import sys

from feedback_store import (
    Correction, CorrectionStatus, Feedback, FeedbackStore, Rating, Reason, ResponseStyle,
    StylePreference, UserProfile, can_propose_correction,
)
from validation import ValidationModule
from dpo_train import DPOTrainingPipeline, ModelRegistry
from dashboard import print_dashboard

sys.path.insert(0, ".")  # permet d'importer minirag.py placé à côté
from minirag import LocalLLM, RAGChatbot, EmbeddingIndex, load_articles  # noqa: E402


REASON_PROMPT = {
    "1": Reason.TOO_GENERAL,
    "2": Reason.LEGALLY_INCORRECT,
    "3": Reason.INCOMPLETE,
    "4": Reason.OFF_TOPIC,
    "5": Reason.OTHER,
}

PROFILE_PROMPT = {
    "1": UserProfile.CITIZEN,
    "2": UserProfile.LAWYER_VERIFIED,
    "3": UserProfile.JURIST_INTERNAL,
    "4": UserProfile.ANONYMOUS,
}

# §5.5 — deux variantes de système-prompt utilisées pour l'A/B de structure
# proposé aux citoyens/anonymes.
STYLE_SYSTEM_PROMPTS = {
    ResponseStyle.PEDAGOGICAL: (
        "Tu es un assistant juridique. Réponds UNIQUEMENT à partir des articles "
        "de loi fournis en contexte. Utilise un langage simple et pédagogique, "
        "avec un exemple concret si utile, comme si tu expliquais à quelqu'un "
        "sans formation juridique. Cite quand même le numéro d'article utilisé."
    ),
    ResponseStyle.TECHNICAL: (
        "Tu es un assistant juridique. Réponds UNIQUEMENT à partir des articles "
        "de loi fournis en contexte. Utilise un langage technique et précis, "
        "structuré, avec référence systématique et rigoureuse au numéro "
        "d'article et à la loi concernée, comme pour un professionnel du droit."
    ),
}


def ask_profile() -> UserProfile:
    print("Profil : 1) Citoyen  2) Avocat vérifié  3) Juriste interne  4) Anonyme")
    choice = input("> ").strip()
    return PROFILE_PROMPT.get(choice, UserProfile.ANONYMOUS)


def collect_feedback(store: FeedbackStore, question: str, answer: str,
                      sources: list[str], profile: UserProfile) -> Feedback | None:
    """§5.1 — pouce haut/bas + motif prédéfini, pour TOUS les profils."""
    print("\nCette réponse est-elle utile ? 1) 👍  2) 👎  (Entrée = ignorer)")
    choice = input("> ").strip()
    if choice not in {"1", "2"}:
        return None

    rating = Rating.UP if choice == "1" else Rating.DOWN
    reason, reason_text = None, None
    if rating == Rating.DOWN:
        print("Motif : 1) Trop générale  2) Juridiquement incorrecte  "
              "3) Incomplète  4) Hors sujet  5) Autre")
        reason = REASON_PROMPT.get(input("> ").strip(), Reason.OTHER)
        if reason == Reason.OTHER:
            reason_text = input("Précisez : ").strip()

    fb = store.add_feedback(Feedback(
        question=question, answer=answer, rating=rating,
        user_profile=profile, reason=reason, reason_text=reason_text, sources=sources,
    ))

    # §5.2 : seuls avocat_verifie / juriste_interne peuvent corriger le fond.
    if rating == Rating.DOWN and can_propose_correction(profile):
        collect_correction(store, fb, profile)
    return fb


def collect_correction(store: FeedbackStore, fb: Feedback, profile: UserProfile) -> None:
    """§5.2 — réservé à avocat_verifie / juriste_interne."""
    print("Souhaitez-vous proposer une réponse corrigée ? (laisser vide pour passer)")
    proposed = input("> ").strip()
    if not proposed:
        return
    correction = Correction(
        feedback_id=fb.id, question=fb.question,
        rejected_answer=fb.answer, proposed_answer=proposed,
        author_profile=profile,
    )
    store.add_correction(correction)
    print(f"Correction enregistrée ({correction.id}), en file d'attente de validation.")


def collect_style_choice(store: FeedbackStore, question: str, answer_a: str, answer_b: str,
                          profile: UserProfile) -> str:
    """
    §5.5 — pour citoyen/anonyme : demande DIRECTEMENT laquelle des deux
    structures de réponse est préférée (façon "réponse A / réponse B" de
    ChatGPT), indépendamment de toute note de satisfaction. Retourne la
    réponse choisie (celle affichée comme "la" réponse à l'utilisateur).
    Ne passe jamais par la validation experte (porte sur la forme, pas le fond).
    """
    print("\nQuelle réponse préférez-vous ? 1) A  2) B  (Entrée = A par défaut)")
    choice = input("> ").strip()
    chosen = "b" if choice == "2" else "a"

    store.add_style_preference(StylePreference(
        question=question,
        answer_a=answer_a, style_a=ResponseStyle.PEDAGOGICAL,
        answer_b=answer_b, style_b=ResponseStyle.TECHNICAL,
        chosen=chosen, user_profile=profile,
    ))
    return answer_a if chosen == "a" else answer_b


def handle_professional_turn(store: FeedbackStore, bot: RAGChatbot, question: str,
                              profile: UserProfile) -> None:
    """Avocat / juriste : une seule réponse affichée, avec droit de correction (§5.2)."""
    result = bot.ask(question)
    print("\n--- Réponse ---")
    print(result["answer"])
    print("\n--- Sources ---")
    for s in result["sources"]:
        print(f"  ({s['score']}) {s['citation']}")
    sources = [s["citation"] for s in result["sources"]]
    collect_feedback(store, question, result["answer"], sources, profile)


def handle_citizen_turn(store: FeedbackStore, index: EmbeddingIndex, llm: LocalLLM,
                         question: str, profile: UserProfile, top_k: int) -> None:
    """Citoyen / anonyme : deux structures générées directement, choix A/B (§5.5)."""
    hits = index.search(question, top_k=top_k)
    hit_articles = [a for a, _score in hits]
    answer_a = llm.generate(question, hit_articles, system_prompt=STYLE_SYSTEM_PROMPTS[ResponseStyle.PEDAGOGICAL])
    answer_b = llm.generate(question, hit_articles, system_prompt=STYLE_SYSTEM_PROMPTS[ResponseStyle.TECHNICAL])

    print("\n--- Réponse A ---")
    print(answer_a)
    print("\n--- Réponse B ---")
    print(answer_b)
    print("\n--- Sources ---")
    for a, score in hits:
        print(f"  ({round(score, 3)}) {a.citation()}")

    chosen_answer = collect_style_choice(store, question, answer_a, answer_b, profile)
    sources = [a.citation() for a in hit_articles]
    collect_feedback(store, question, chosen_answer, sources, profile)


def cmd_chat(args: argparse.Namespace) -> None:
    print(f"Chargement des articles depuis {args.data} ...")
    articles = load_articles(args.data)
    index = EmbeddingIndex(args.embed_model)
    index.build(articles)
    llm = LocalLLM(args.llm_model)
    bot = RAGChatbot(index, llm, top_k=args.top_k)
    store = FeedbackStore(args.db)

    profile = ask_profile()
    can_correct = can_propose_correction(profile)
    print("\nPrêt. Tapez votre question ('quit' pour sortir).\n")
    while True:
        q = input("Question> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue

        if can_correct:
            handle_professional_turn(store, bot, q, profile)
        else:
            handle_citizen_turn(store, index, llm, q, profile, args.top_k)
        print()
    store.close()


def cmd_review(args: argparse.Namespace) -> None:
    """File d'attente de l'expert validateur — §4, §5.3."""
    print(f"Chargement des articles depuis {args.data} ...")
    articles = load_articles(args.data)
    index = EmbeddingIndex(args.embed_model)
    index.build(articles)
    store = FeedbackStore(args.db)
    validator = ValidationModule(index, store)

    pending_rows = store.pending_corrections()
    print(f"Corrections en file d'attente (en_attente + expert_requis) : {len(pending_rows)}")
    if not pending_rows:
        print("Rien à faire pour le moment. Rappel : seuls avocat_verifie / juriste_interne "
              "peuvent soumettre des corrections (§5.2) — vérifiez que la base "
              f"'{args.db}' contient bien des corrections de ces profils.")
        store.close()
        return

    n_shown = 0
    for row in pending_rows:
        correction = Correction(
            id=row["id"], feedback_id=row["feedback_id"], question=row["question"],
            rejected_answer=row["rejected_answer"], proposed_answer=row["proposed_answer"],
            author_profile=UserProfile(row["author_profile"]),
            status=CorrectionStatus(row["status"]), timestamp=row["timestamp"],
        )
        if correction.status == CorrectionStatus.PENDING:
            correction = validator.validate(correction)
            print(f"\n[{correction.id}] validation automatique -> {correction.status.value} "
                  f"(confiance={correction.confidence_score}, consensus={correction.consensus_count}, "
                  f"RAG={correction.rag_check})")
        if correction.status != CorrectionStatus.NEEDS_EXPERT:
            continue  # auto-validée (ou déjà traitée) : rien à faire ici

        n_shown += 1
        print("\n" + "=" * 56)
        print(f"Correction {correction.id}  (confiance={correction.confidence_score}, "
              f"consensus={correction.consensus_count}, RAG={correction.rag_check})")
        print(f"Question   : {correction.question}")
        print(f"Rejetée    : {correction.rejected_answer}")
        print(f"Proposée   : {correction.proposed_answer}")
        print("Décision : 1) Valider  2) Rejeter  3) Reformuler  (Entrée = passer)")
        choice = input("> ").strip()
        if choice == "1":
            validator.expert_decide(correction, "valide", args.validator_role)
        elif choice == "2":
            validator.expert_decide(correction, "rejete", args.validator_role)
        elif choice == "3":
            reformulated = input("Nouvelle réponse : ").strip()
            validator.expert_decide(correction, "reformule", args.validator_role, reformulated)
    if n_shown == 0:
        print("\nToutes les corrections en attente ont été auto-validées "
              "(confiance + consensus suffisants) — aucune n'a nécessité votre revue.")
    store.close()


def cmd_train(args: argparse.Namespace) -> None:
    """§5.4, §7 étapes 5-7 — déclenche un cycle DPO si assez de données."""
    store = FeedbackStore(args.db)
    pipeline = DPOTrainingPipeline(base_model=args.llm_model)
    if not pipeline.should_trigger(store, min_examples=args.min_examples):
        print(f"Pas assez de corrections validées (< {args.min_examples}). Cycle annulé.")
        store.close()
        return
    mv = pipeline.run_cycle(store, min_examples=args.min_examples, epochs=args.epochs)
    print(f"Nouvelle version entraînée : v{mv.version} -> {mv.path}")
    print("Statut 'staged' : à évaluer (comparaison A/B, §7-étape 8) avant promotion.")
    store.close()


def cmd_promote(args: argparse.Namespace) -> None:
    ModelRegistry().promote(args.version)
    print(f"Version v{args.version} promue en production.")


def cmd_dashboard(args: argparse.Namespace) -> None:
    store = FeedbackStore(args.db)
    print_dashboard(store)
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot juridique + boucle feedback/DPO (E-Tafakna)")
    parser.add_argument("--db", default="feedback.db")
    parser.add_argument("--data", default="articles")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--top-k", type=int, default=3)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("chat")

    p_review = sub.add_parser("review")
    p_review.add_argument("--validator-role", default="expert_juriste")

    p_train = sub.add_parser("train")
    p_train.add_argument("--min-examples", type=int, default=20)
    p_train.add_argument("--epochs", type=int, default=1)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("version", type=int)

    sub.add_parser("dashboard")

    args = parser.parse_args()
    {
        "chat": cmd_chat,
        "review": cmd_review,
        "train": cmd_train,
        "promote": cmd_promote,
        "dashboard": cmd_dashboard,
    }[args.command](args)


if __name__ == "__main__":
    main()
