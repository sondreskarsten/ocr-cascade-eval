from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import pandas as pd
    from transformers import pipeline

    df = pd.read_csv(fx["table.csv"]).astype(str)
    pipe = pipeline("table-question-answering",
                    model="google/tapas-base-finetuned-wtq")
    questions = [
        "What is the value of Årsresultat in 2024?",
        "How much was Skattekostnad in 2023?",
        "Sum of Driftsresultat across years?",
    ]
    answers = [pipe(table=df, query=q) for q in questions]
    return {"checkpoint": "google/tapas-base-finetuned-wtq",
            "table_columns": list(df.columns),
            "table_n_rows": len(df),
            "questions": questions,
            "answers": [{k: str(v) for k, v in a.items()} for a in answers]}


if __name__ == "__main__":
    run_with_metrics("tapas", main)
