from shared import for_each_pdf, run_with_metrics
import pandas as pd


def _extract_first_table(pdf_path):
    import camelot
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        for t in tables:
            df = t.df
            if df.shape[0] >= 3 and df.shape[1] >= 2:
                return df.astype(str)
    except Exception:
        pass
    return None


def main():
    from transformers import pipeline
    pipe = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
    questions = ["What is årsresultat?", "Sum kostnader?", "Driftsresultat?"]

    def per_pdf(pdf_id, b):
        df = _extract_first_table(b["pdf_ocr"])
        if df is None:
            return {"status": "no_table_extracted"}
        answers = []
        for q in questions:
            try:
                a = pipe(table=df, query=q)
                answers.append({"q": q, "answer": str(a.get("answer", a))})
            except Exception as e:
                answers.append({"q": q, "error": f"{type(e).__name__}: {e}"})
        return {"table_shape": list(df.shape), "table_first_row": df.iloc[0].tolist(),
                "questions_and_answers": answers}

    return {"checkpoint": "google/tapas-base-finetuned-wtq", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("tapas", main)
