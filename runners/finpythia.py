from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    samples = json.loads(open(fx["samples.json"]).read())
    ckpt = "amazon/FinPythia-1.4B"
    info = {"checkpoint": ckpt}
    try:
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32, device_map="cpu")
        model.eval()
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return info
    prompt = samples["norwegian_finance_sentence"] + " The sentiment is"
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    info["completion"] = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return info


if __name__ == "__main__":
    run_with_metrics("finpythia", main)
