from shared import for_each_pdf, run_with_metrics


def main():
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    train_examples = [
        {"text": "Skattekostnad", "label": 0},
        {"text": "Betalbar skatt", "label": 0},
        {"text": "Utsatt skatt", "label": 0},
        {"text": "Endring utsatt skatt", "label": 0},
        {"text": "Sum driftsinntekter", "label": 1},
        {"text": "Annen driftsinntekt", "label": 1},
        {"text": "Salgsinntekter", "label": 1},
        {"text": "Driftsinntekter", "label": 1},
    ]
    train_ds = Dataset.from_list(train_examples)
    model = SetFitModel.from_pretrained("NbAiLab/nb-sbert-base", labels=["tax", "revenue"])
    args = TrainingArguments(num_epochs=1, batch_size=4, num_iterations=10)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=train_ds,
                      column_mapping={"text": "text", "label": "label"})
    trainer.train()

    def per_pdf(pdf_id, b):
        lines = []
        for ln in b["full_text"].splitlines():
            ln = ln.strip()
            if 4 <= len(ln) <= 50 and ln[0:1].isalpha() and ln not in lines: lines.append(ln)
            if len(lines) >= 20: break
        if not lines:
            return {"n_inputs": 0}
        preds = model.predict(lines)
        results = [{"line": ln, "pred": p} for ln, p in zip(lines, list(preds))]
        return {"n_inputs": len(lines), "predictions": results}

    return {"backbone": "NbAiLab/nb-sbert-base", "n_train_examples": len(train_examples),
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("setfit", main)
