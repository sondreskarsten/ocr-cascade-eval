from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # No FinGPT-Pythia variant exists. Test EleutherAI/pythia-1.4b base on financial sentiment.
    base = "EleutherAI/pythia-1.4b"
    info = {"intent": "FinGPT-Pythia variant",
            "note": "No FinGPT-Pythia adapter on HF Hub; testing base Pythia-1.4b directly"}
    try:
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")
        model.eval()
    except Exception as e:
        return {**info, "status": "error",
                "msg": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        first_lines = "\n".join(b["full_text"].splitlines()[:6])
        prompt = (f"Sentiment of this Norwegian financial text (negative/neutral/positive)?\n"
                  f"Text: {first_lines}\nAnswer:")
        try:
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
            ans = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return {"answer": ans.strip()}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    return {**info, "checkpoint": base, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("finpythia", main)
