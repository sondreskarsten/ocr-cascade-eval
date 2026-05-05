from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())
    from transformers import pipeline
    candidates = ["NbAiLab/nb-bert-base-ner", "saattrupdan/nbailab-base-ner-scandi"]
    chosen, log = None, []
    pipe = None
    for c in candidates:
        try:
            pipe = pipeline("token-classification", model=c, aggregation_strategy="simple")
            chosen = c
            break
        except Exception as e:
            log.append({c: f"{type(e).__name__}: {e}"})
    if pipe is None:
        return {"status": "error", "lookup_log": log}
    ents = pipe(samples["ner_text"])
    return {"checkpoint": chosen, "lookup_log": log,
            "text": samples["ner_text"],
            "entities": [{k: float(v) if hasattr(v, "item") else v for k, v in e.items()} for e in ents]}


if __name__ == "__main__":
    run_with_metrics("nb_ner", main)
