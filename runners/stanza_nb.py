from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())
    info = {"library": "stanza"}
    try:
        import stanza
        stanza.download("nb", verbose=False)
        nlp = stanza.Pipeline("nb", processors="tokenize,ner", verbose=False)
        doc = nlp(samples["ner_text"])
        ents = [{"text": e.text, "type": e.type} for e in doc.entities]
        info["text"] = samples["ner_text"]
        info["entities"] = ents
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("stanza_nb", main)
