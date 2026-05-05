from shared import for_each_pdf, run_with_metrics


def main():
    import spacy
    nlp = spacy.load("nb_core_news_lg")

    def per_pdf(pdf_id, b):
        doc = nlp(b["full_text"][:4000])
        ents = [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                for e in doc.ents]
        return {"input_chars": len(b["full_text"][:4000]),
                "n_entities": len(ents), "entities": ents[:50]}

    return {"checkpoint": "nb_core_news_lg", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("spacy_nb", main)
