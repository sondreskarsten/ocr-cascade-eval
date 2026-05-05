from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    samples = json.loads(open(fx["samples.json"]).read())
    base = "meta-llama/Llama-2-7b-hf"
    adapter = "FinGPT/fingpt-mt_llama2-7b_lora"

    try:
        tok = AutoTokenizer.from_pretrained(base, use_fast=False)
    except Exception as e:
        return {"status": "error",
                "note": f"Llama-2-7b-hf is gated on HF; cannot load without HF token. {e}",
                "checkpoint": adapter,
                "expected_size_gb": 13.5}

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    prompt = (
        "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}\n"
        f"Input: {samples['norwegian_finance_sentence']}\n"
        "Answer: "
    )
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    ans = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"checkpoint": f"{adapter} on {base}",
            "prompt": prompt, "answer": ans.strip()}


if __name__ == "__main__":
    run_with_metrics("fingpt", main)
