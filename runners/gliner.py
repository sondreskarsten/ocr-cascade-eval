from shared import for_each_pdf, run_with_metrics


def main():
    from gliner import GLiNER
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    labels = ["organisasjon", "person", "valuta", "tall", "rolle",
              "regnskapspost", "dato", "sted"]

    def per_pdf(pdf_id, b):
        text = b["full_text"][:4000]
        ents = model.predict_entities(text, labels)
        return {"input_chars": len(text), "labels": labels, "n_entities": len(ents),
                "entities": ents[:30]}

    return {"checkpoint": "urchade/gliner_multi-v2.1",
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("gliner", main)
