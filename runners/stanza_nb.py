from shared import for_each_pdf, run_with_metrics


def main():
    import stanza
    try:
        stanza.download("nb", verbose=False)
        nlp = stanza.Pipeline(lang="nb", processors="tokenize,ner", verbose=False)
    except Exception as e:
        return {"status": "error", "msg": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        doc = nlp(b["full_text"][:4000])
        ents = [{"text": e.text, "type": e.type, "start": e.start_char, "end": e.end_char}
                for s in doc.sentences for e in s.ents]
        return {"n_entities": len(ents), "entities": ents[:50]}

    return {"checkpoint": "stanza nb", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("stanza_nb", main)
