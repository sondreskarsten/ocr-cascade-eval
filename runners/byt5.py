from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    samples = json.loads(open(fx["samples.json"]).read())
    ckpt = "google/byt5-small"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt)
    model.eval()
    inputs = tok([samples["norwegian_label"]], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    text = tok.batch_decode(out, skip_special_tokens=True)[0]
    return {"checkpoint": ckpt,
            "input": samples["norwegian_label"],
            "output_pretrained_only": text,
            "note": "ByT5-small base — character-level T5 with no task fine-tuning. Output reflects vanilla LM behaviour."}


if __name__ == "__main__":
    run_with_metrics("byt5", main)
