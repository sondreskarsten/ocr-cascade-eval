from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, subprocess, sys
    samples = json.loads(open(fx["samples.json"]).read())
    info = {"library": "spacy", "model": "nb_core_news_lg"}
    try:
        import spacy
        try:
            nlp = spacy.load("nb_core_news_lg")
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", "nb_core_news_lg"],
                           capture_output=True, timeout=300)
            nlp = spacy.load("nb_core_news_lg")
        doc = nlp(samples["ner_text"])
        ents = [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                for e in doc.ents]
        info["text"] = samples["ner_text"]
        info["entities"] = ents
        info["n_tokens"] = len(doc)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("spacy_nb", main)
