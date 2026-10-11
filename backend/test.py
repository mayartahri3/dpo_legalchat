import sqlite3
conn = sqlite3.connect("feedback.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("""
    SELECT id, question, proposed_answer, final_answer, rejected_answer, status
    FROM corrections
    WHERE status IN ('validee', 'auto_validee', 'reformulee')
""").fetchall()

print(f"Total validated: {len(rows)}")
degenerate = [r for r in rows if (r["final_answer"] or r["proposed_answer"] or "").strip() == (r["rejected_answer"] or "").strip()]
print(f"Degenerate (chosen == rejected): {len(degenerate)}")
for r in degenerate:
    print(" -", r["id"], "|", r["question"][:60])