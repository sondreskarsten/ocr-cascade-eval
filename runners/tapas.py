from shared import for_each_pdf, run_with_metrics, fetch_fixture


def main():
    fx = fetch_fixture()
    import pandas as pd
    from transformers import pipeline

    df = pd.read_csv(fx["table.csv"]).astype(str)
    pipe = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")

    questions = [
        "What is the value of Årsresultat in 2024?",
        "How much was Skattekostnad in 2023?",
        "What is the value of Sum kostnader in 2024?",
        "What is Driftsresultat in 2023?",
    ]
    answers = []
    for q in questions:
        try:
            a = pipe(table=df, query=q)
            answers.append({"q": q, "a": {k: str(v) for k, v in a.items()}})
        except Exception as e:
            answers.append({"q": q, "error": f"{type(e).__name__}: {e}"})

    def per_pdf(pdf_id, b):
        return {"note": "TaPas reads from static fixture table.csv (same for both PDFs)",
                "table_chars": len(b["full_text"]) if b.get("full_text") else 0}

    return {"checkpoint": "google/tapas-base-finetuned-wtq",
            "table_columns": list(df.columns),
            "table_n_rows": len(df),
            "questions_and_answers": answers,
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("tapas", main)
