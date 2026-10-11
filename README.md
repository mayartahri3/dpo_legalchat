# E-Tafakna: Legal RAG Chatbot with a Feedback Loop

A chatbot that answers questions about Tunisian law using the relevant law articles, and improves from expert feedback.

**How it works:** question → retrieve articles → local LLM answers with citations → users give feedback → lawyers propose corrections → experts validate → validated corrections fine-tune the model (DPO / GRPO) → new models are tested on a frozen test set before promotion.

## Project structure

```
backend/     FastAPI API, retrieval, validation, training and evaluation scripts
frontend/    React (Vite) web interface
```

Main backend files:

| File | What it does |
|---|---|
| `api.py` | The API (chat, feedback, review, training, models) |
| `minirag.py` | Retrieval (FAISS) and answer generation |
| `feedback_store.py` | SQLite database for feedback and corrections |
| `validation.py` | Scores and routes corrections to experts |
| `dataset_builder.py` | Builds the training pairs and the train/test split |
| `dpo_train.py`, `grpo_train.py`, `reward_model.py` | Training methods |
| `eval_metrics.py` | Evaluation of model versions on the test set |
| `export_for_colab.py`, `colab_train.py`, `import_colab_model.py` | Train on Google Colab, import results locally |

## Run it locally

**Backend**

```bash
cd backend
pip install -r requirements.txt
# put the law article .json files in backend/articles/
uvicorn api:app --reload --port 8000
```

Settings are in `backend/.env` (articles folder, models, `TOP_K`, ...).

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

The API address is set in `frontend/.env` (`VITE_API_URL=http://localhost:8000`).

## Train on Google Colab

1. **Local:** freeze the test set once, then export the data.
   ```bash
   cd backend
   python eval_metrics.py --make-split
   python export_for_colab.py
   ```
2. **Colab (GPU runtime):** upload `feedback_export.zip` and the backend `.py` files, install the dependencies, then run:
   ```bash
   python colab_train.py --export-zip feedback_export.zip --method all --epochs 3
   ```
   Download `trained_model_export.zip` when it finishes.
3. **Local:** import the models.
   ```bash
   python -m zipfile -e trained_model_export.zip trained_model_export_colab
   python import_colab_model.py --export-dir ./trained_model_export_colab
   ```

Tip: use `--fast-test` first to check that everything runs.

## Evaluate and promote

```bash
python eval_metrics.py --split test --write-registry
```

This compares each version with the base model on questions it never saw. Then promote a version in the **Entraînement** page of the UI and restart the API.

## Notes

- Only lawyers and internal jurists can propose corrections; citizens choose between two answer styles.
- PPO is not used (the installed `trl` no longer includes it).
- With few validated corrections the test set is small, so treat results as indicative.