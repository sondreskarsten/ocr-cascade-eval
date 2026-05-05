from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    samples = json.loads(open(fx["samples.json"]).read())
    ckpt = "ltg/nort5-base"
    try:
        tok = T5Tokenizer.from_pretrained(ckpt, trust_remote_code=True)
        model = T5ForConditionalGeneration.from_pretrained(ckpt, trust_remote_code=True)
    except Exception as e:
        return {"status": "error", "checkpoint": ckpt,
                "error": f"{type(e).__name__}: {e}"}
    model.eval()
    inputs = tok([samples["norwegian_label"]], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    text = tok.batch_decode(out, skip_special_tokens=True)[0]
    return {"checkpoint": ckpt, "input": samples["norwegian_label"], "output": text}


if __name__ == "__main__":
    run_with_metrics("nort5", main)
