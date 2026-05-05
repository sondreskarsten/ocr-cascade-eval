from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import pandas as pd, torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    ckpt = "neulab/omnitab-large-finetuned-wtq"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    model.eval()

    df = pd.read_csv(fx["table.csv"]).astype(str)
    questions = [
        "What is the value of Årsresultat in 2024?",
        "How much was Skattekostnad in 2023?",
    ]
    answers = []
    for q in questions:
        enc = tok(table=df, query=q, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=32)
        ans = tok.batch_decode(out, skip_special_tokens=True)[0]
        answers.append({"q": q, "a": ans})
    return {"checkpoint": ckpt, "questions_and_answers": answers}


if __name__ == "__main__":
    run_with_metrics("omnitab", main)
