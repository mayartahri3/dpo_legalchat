import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const client = axios.create({ baseURL: BASE_URL });

export const api = {
  chat: (question, profile, user_id) =>
    client.post("/api/chat", { question, profile, user_id }).then((r) => r.data),

  history: (user_id) => client.get("/api/history", { params: { user_id } }).then((r) => r.data),

  sendFeedback: ({ question, answer, rating, profile, reason, reason_text, sources }) =>
    client
      .post("/api/feedback", { question, answer, rating, profile, reason, reason_text, sources })
      .then((r) => r.data),

  sendCorrection: ({ feedback_id, question, rejected_answer, proposed_answer, profile }) =>
    client
      .post("/api/correction", { feedback_id, question, rejected_answer, proposed_answer, profile })
      .then((r) => r.data),

  sendStylePreference: ({ question, answer_a, answer_b, chosen, profile, turn_id }) =>
    client
      .post("/api/style-preference", { question, answer_a, answer_b, chosen, profile, turn_id })
      .then((r) => r.data),

  reviewPending: () => client.get("/api/review/pending").then((r) => r.data),

  reviewDecide: ({ correction_id, decision, validator_role, reformulated_answer }) =>
    client
      .post("/api/review/decide", { correction_id, decision, validator_role, reformulated_answer })
      .then((r) => r.data),

  dashboard: () => client.get("/api/dashboard").then((r) => r.data),

  train: ({ min_examples, epochs, method = "dpo" }) =>
    client.post("/api/train", { min_examples, epochs, method }).then((r) => r.data),

  compareModels: ({ versions, n_questions = 15 }) =>
    client.post("/api/models/compare", { versions, n_questions }).then((r) => r.data),

  rewardModelHistory: () => client.get("/api/reward-model/history").then((r) => r.data),

  seedDemoData: (n) => client.post("/api/seed-demo-data", { n }).then((r) => r.data),

  deleteSeedDemoData: () => client.delete("/api/seed-demo-data").then((r) => r.data),

  trainStatus: (job_id) => client.get(`/api/train/status/${job_id}`).then((r) => r.data),

  trainJobs: () => client.get("/api/train/jobs").then((r) => r.data),

  trainStats: () => client.get("/api/train/stats").then((r) => r.data),

  triggerN8n: ({ min_examples, epochs }) =>
    client.post("/api/train/trigger-n8n", { min_examples, epochs }).then((r) => r.data),

  importModel: (file, { base_model, n_training_pairs, dataset_version, method } = {}) => {
  const form = new FormData();
  form.append("file", file);
  if (base_model) form.append("base_model", base_model);
  if (n_training_pairs != null) form.append("n_training_pairs", n_training_pairs);
  if (dataset_version != null) form.append("dataset_version", dataset_version);
  if (method) form.append("method", method);   // NEW
  return client
    .post("/api/models/import", form, { headers: { "Content-Type": "multipart/form-data" } })
    .then((r) => r.data);
},

  modelsHistory: () => client.get("/api/models/history").then((r) => r.data),

  promote: (version) => client.post(`/api/promote/${version}`).then((r) => r.data),

  listArticles: () => client.get("/api/articles").then((r) => r.data),

  searchArticles: (q, top_k = 30) =>
    client.get("/api/articles/search", { params: { q, top_k } }).then((r) => r.data),

  health: () => client.get("/api/health").then((r) => r.data),
};

export default api;