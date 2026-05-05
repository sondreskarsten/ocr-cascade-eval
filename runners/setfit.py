from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
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
    eval_examples = [
        {"text": "Resultatført skatt", "label": 0},
        {"text": "Driftsinntekter totalt", "label": 1},
    ]
    train_ds = Dataset.from_list(train_examples)
    eval_ds = Dataset.from_list(eval_examples)
    model = SetFitModel.from_pretrained("NbAiLab/nb-sbert-base", labels=["tax", "revenue"])

    args = TrainingArguments(num_epochs=1, batch_size=4, num_iterations=20)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      column_mapping={"text": "text", "label": "label"})
    trainer.train()
    metrics = trainer.evaluate()
    preds = model.predict(["Resultatført skatt på ordinært resultat",
                           "Sum salgsinntekter 2024",
                           "Annen finansinntekt"])
    return {"backbone": "NbAiLab/nb-sbert-base",
            "n_train": len(train_examples),
            "metrics": dict(metrics) if hasattr(metrics, 'items') else metrics,
            "predictions_on_unseen": list(preds)}


if __name__ == "__main__":
    run_with_metrics("setfit", main)
