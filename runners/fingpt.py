from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    base_candidates = [
        "NousResearch/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-hf",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
    adapter = "FinGPT/fingpt-mt_llama2-7b_lora"
    base_chosen, log = None, []
    for b in base_candidates:
        try:
            tok = AutoTokenizer.from_pretrained(b, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(b, torch_dtype=torch.float32, device_map="cpu")
            base_chosen = b
            break
        except Exception as e:
            log.append({b: f"{type(e).__name__}: {str(e)[:200]}"})
    if base_chosen is None:
        return {"status": "error", "tried": base_candidates,
                "log": log,
                "note": "all base llama2-7b candidates inaccessible (gated/missing)"}

    try:
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    except Exception as e:
        return {"status": "error", "base": base_chosen,
                "adapter_error": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        first_lines = "\n".join(b["full_text"].splitlines()[:10])
        prompt = (f"Instruction: What is the sentiment of this Norwegian financial text? "
                  f"Choose: negative/neutral/positive.\nInput: {first_lines}\nAnswer: ")
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        ans = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"prompt_chars": len(prompt), "answer": ans.strip()}

    return {"base": base_chosen, "adapter": adapter, "log": log,
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("fingpt", main)
