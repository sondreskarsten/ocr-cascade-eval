from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from gliner import GLiNER

    samples = json.loads(open(fx["samples.json"]).read())
    candidates = ["fastino/GLiNER2", "knowledgator/gliner-multitask-large-v0.5",
                  "urchade/gliner_large-v2.5"]
    chosen, log = None, []
    for c in candidates:
        try:
            model = GLiNER.from_pretrained(c)
            chosen = c
            break
        except Exception as e:
            log.append({c: f"{type(e).__name__}: {e}"})
    if chosen is None:
        return {"status": "error", "lookup_log": log}
    text = samples["ner_text"]
    labels = samples["ner_labels"]
    ents = model.predict_entities(text, labels)
    return {"checkpoint": chosen, "lookup_log": log,
            "text": text, "labels": labels, "entities": ents}


if __name__ == "__main__":
    run_with_metrics("gliner2", main)
