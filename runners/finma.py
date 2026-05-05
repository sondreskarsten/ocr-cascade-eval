from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    samples = json.loads(open(fx["samples.json"]).read())
    ckpt = "ChanceFocus/finma-7b-nlp"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    prompt = (
        "Analyze the sentiment of this statement extracted from a financial news article. "
        "Provide your answer as either negative, positive, or neutral.\n"
        f"Text: {samples['norwegian_finance_sentence']}\nAnswer:"
    )
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    ans = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"checkpoint": ckpt, "prompt": prompt, "answer": ans.strip()}


if __name__ == "__main__":
    run_with_metrics("finma", main)
