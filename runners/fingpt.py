from shared import for_each_pdf, run_with_metrics


def main():
    import os, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    bases = ["NousResearch/Llama-2-7b-hf", "meta-llama/Llama-2-7b-hf"]
    base = None
    err = None
    for cand in bases:
        try:
            tok = AutoTokenizer.from_pretrained(cand, use_fast=False)
            base = cand; break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    if base is None:
        return {"status": "error", "note": "no non-gated Llama2 base accessible", "last_error": err}

    adapter = "FinGPT/fingpt-mt_llama2-7b_lora"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def per_pdf(pdf_id, b):
        excerpt = b["full_text"][:1500]
        prompt = (
            f"Instruction: Determine whether the financial situation reflects negative, neutral or positive performance. "
            f"Choose from negative/neutral/positive only.\n"
            f"Input: {excerpt}\nAnswer: "
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1500)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        ans = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"input_chars": len(excerpt), "answer": ans.strip()}

    return {"base": base, "adapter": adapter, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("fingpt", main)
