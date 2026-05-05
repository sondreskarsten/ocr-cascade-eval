from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ckpt = "FinGPT/fingpt-mt_pythia-1.4b_lora"
    base = "EleutherAI/pythia-1.4b"
    try:
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ckpt)
        model.eval()
    except Exception as e:
        return {"status": "error", "msg": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        excerpt = b["full_text"][:1500]
        prompt = f"Instruction: Sentiment of this text? (negative/neutral/positive)\nInput: {excerpt}\nAnswer:"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1500)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        ans = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"answer": ans.strip()}

    return {"base": base, "adapter": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("finpythia", main)
