from shared import for_each_pdf, run_with_metrics


def main():
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    train_examples = [
        # revenue / inntekter
        ("Salgsinntekt", "revenue"), ("Sum inntekter", "revenue"),
        ("Annen driftsinntekt", "revenue"), ("Driftsinntekter", "revenue"),
        ("Andre inntekter", "revenue"), ("Sum driftsinntekter", "revenue"),
        # cost / kostnader
        ("Sum kostnader", "cost"), ("Annen driftskostnad", "cost"),
        ("Lønnskostnad", "cost"), ("Avskrivninger", "cost"),
        ("Varekostnad", "cost"), ("Sum driftskostnad", "cost"),
        # equity / egenkapital
        ("Sum egenkapital", "equity"), ("Aksjekapital", "equity"),
        ("Opptjent egenkapital", "equity"), ("Innskutt egenkapital", "equity"),
        ("Annen egenkapital", "equity"), ("Egenkapital 31.12", "equity"),
        # debt / gjeld
        ("Sum gjeld", "debt"), ("Langsiktig gjeld", "debt"),
        ("Kortsiktig gjeld", "debt"), ("Leverandørgjeld", "debt"),
        ("Skyldig offentlig avgift", "debt"), ("Annen kortsiktig gjeld", "debt"),
        # metadata / non-financial line
        ("Brønnøysundregistrene", "meta"), ("Organisasjonsnummer", "meta"),
        ("Foretaksnavn", "meta"), ("Forretningsadresse", "meta"),
        ("Journalnummer", "meta"), ("Aksjeselskap", "meta"),
    ]
    texts, labels = zip(*train_examples)
    train_ds = Dataset.from_dict({"text": list(texts), "label": list(labels)})

    model = SetFitModel.from_pretrained("NbAiLab/nb-sbert-base")
    args = TrainingArguments(num_epochs=1, batch_size=8)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    def per_pdf(pdf_id, b):
        seen = []
        for ln in b["full_text"].splitlines():
            ln = ln.strip()
            if 4 <= len(ln) <= 60 and not ln.replace(" ","").isdigit():
                if ln not in seen: seen.append(ln)
            if len(seen) >= 25: break
        if not seen: return {"n_inputs": 0, "predictions": []}
        preds = model.predict(seen)
        out = [{"line": s, "pred": str(p)} for s, p in zip(seen, preds)]
        # Distribution
        from collections import Counter
        dist = Counter(p["pred"] for p in out)
        return {"n_inputs": len(seen), "predictions": out, "label_distribution": dict(dist)}

    return {"backbone": "NbAiLab/nb-sbert-base",
            "n_train_examples": len(train_examples),
            "labels": sorted(set(labels)),
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("setfit", main)
