from shared import for_each_pdf, run_with_metrics


def main():
    from transformers import pipeline
    try:
        pipe = pipeline("ner", model="saattrupdan/nbailab-base-ner-scandi", aggregation_strategy="simple")
        task = "ner"
    except Exception:
        try:
            pipe = pipeline("fill-mask", model="saattrupdan/nbailab-base-ner-scandi")
            task = "fill-mask"
        except Exception as e:
            return {"status": "error", "msg": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        text = b["full_text"][:2000]
        if task == "ner":
            res = pipe(text)
            return {"task": "ner", "n_entities": len(res),
                    "entities": [{"word": x.get("word"), "type": x.get("entity_group"),
                                  "score": round(float(x.get("score",0)), 3)} for x in res[:30]]}
        else:
            sample = "Selskapet har en betydelig [MASK] i norske banker."
            res = pipe(sample)
            return {"task": "fill-mask", "input": sample,
                    "predictions": [{"token": r["token_str"], "score": round(r["score"], 3)} for r in res[:5]]}

    return {"checkpoint": "saattrupdan/nbailab-base-ner-scandi", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("nb_ner", main)
