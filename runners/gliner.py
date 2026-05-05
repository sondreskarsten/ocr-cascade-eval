from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from gliner import GLiNER

    samples = json.loads(open(fx["samples.json"]).read())
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    text = samples["ner_text"]
    labels = samples["ner_labels"]
    ents = model.predict_entities(text, labels)
    return {"checkpoint": "urchade/gliner_multi-v2.1",
            "text": text, "labels": labels, "entities": ents}


if __name__ == "__main__":
    run_with_metrics("gliner", main)
