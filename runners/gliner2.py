from shared import for_each_pdf, run_with_metrics


def main():
    from gliner import GLiNER
    candidates = ["urchade/gliner_large-v2.5", "knowledgator/gliner-multitask-large-v0.5",
                  "urchade/gliner_multi-v2.1"]
    model = chosen = None
    log = []
    for c in candidates:
        try:
            model = GLiNER.from_pretrained(c); chosen = c; break
        except Exception as e:
            log.append({c: f"{type(e).__name__}: {e}"})
    if model is None:
        return {"status": "error", "lookup_log": log}
    labels = ["organisasjon", "person", "valuta", "tall", "rolle",
              "regnskapspost", "dato", "sted"]

    def per_pdf(pdf_id, b):
        text = b["full_text"][:4000]
        ents = model.predict_entities(text, labels)
        return {"input_chars": len(text), "labels": labels, "n_entities": len(ents),
                "entities": ents[:30]}

    return {"checkpoint": chosen, "lookup_log": log, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("gliner2", main)
