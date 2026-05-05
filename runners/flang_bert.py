from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from transformers import pipeline
    samples = json.loads(open(fx["samples.json"]).read())
    candidates = ["SALT-NLP/FLANG-BERT", "yiyanghkust/finbert-pretrain"]
    pipe, chosen, log = None, None, []
    for c in candidates:
        try:
            pipe = pipeline("fill-mask", model=c)
            chosen = c
            break
        except Exception as e:
            log.append({c: f"{type(e).__name__}: {e}"})
    if pipe is None:
        return {"status": "error", "lookup_log": log,
                "note": "FLANG-BERT not on HF Hub under that exact id"}
    out = pipe("The bond yield [MASK] last quarter.", top_k=5)
    return {"checkpoint": chosen, "fill_mask_demo": out}


if __name__ == "__main__":
    run_with_metrics("flang_bert", main)
