from shared import for_each_pdf, run_with_metrics
import re


def main():
    from transformers import pipeline
    pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True, max_length=512)

    def per_pdf(pdf_id, b):
        sents = re.split(r"(?<=[.!?])\s+", b["full_text"])
        sents = [s.strip() for s in sents if 20 <= len(s.strip()) <= 300]
        sents = sents[:25]
        preds = []
        for s in sents:
            try:
                p = pipe(s)[0]
                top = max(p, key=lambda x: x["score"])
                preds.append({"text": s[:200], "label": top["label"], "score": round(top["score"], 3),
                               "all_scores": [{"label": x["label"], "score": round(x["score"], 3)} for x in p]})
            except Exception as e:
                preds.append({"text": s[:200], "error": f"{type(e).__name__}: {e}"})
        return {"n_sentences": len(sents), "predictions": preds}

    return {"checkpoint": "ProsusAI/finbert",
            "trained_on": "English-only finance text",
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("finbert_prosus", main)
