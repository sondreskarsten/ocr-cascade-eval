from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from transformers import pipeline

    samples = json.loads(open(fx["samples.json"]).read())
    candidates = ["yiyanghkust/finbert-fls", "ProsusAI/finbert"]
    chosen, err_log = None, []
    for c in candidates:
        try:
            pipe = pipeline("text-classification", model=c, top_k=None)
            chosen = c
            break
        except Exception as e:
            err_log.append({c: f"{type(e).__name__}: {e}"})

    if chosen is None:
        return {"status": "error", "error_log": err_log,
                "note": "FinBERT2 lookup — no published HF id matched"}

    nor_sentence = samples["norwegian_finance_sentence"]
    eng_sentence = "FARBOSS AS reported a strong annual result driven by increased financial income."
    return {
        "checkpoint": chosen,
        "lookup_log": err_log,
        "note": "FinBERT2 not officially published on HF Hub under that name; using FLS variant as proxy.",
        "norwegian_pred": pipe(nor_sentence),
        "english_pred": pipe(eng_sentence),
    }


if __name__ == "__main__":
    run_with_metrics("finbert2", main)
