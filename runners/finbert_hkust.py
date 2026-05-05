from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from transformers import pipeline

    samples = json.loads(open(fx["samples.json"]).read())
    pipe = pipeline("text-classification", model="yiyanghkust/finbert-tone", top_k=None)

    nor_sentence = samples["norwegian_finance_sentence"]
    eng_sentence = "FARBOSS AS reported a strong annual result of 241,101 NOK in 2024, a major improvement from 168,986 NOK the previous year, driven by increased financial income."
    return {
        "norwegian_input": nor_sentence,
        "norwegian_pred": pipe(nor_sentence),
        "english_input": eng_sentence,
        "english_pred": pipe(eng_sentence),
        "checkpoint": "yiyanghkust/finbert-tone",
        "trained_on": "English Analyst report sentences",
    }


if __name__ == "__main__":
    run_with_metrics("finbert_hkust", main)
