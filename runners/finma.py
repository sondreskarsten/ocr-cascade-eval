from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    candidates = ["TheFinAI/finma-7b-nlp", "ChanceFocus/finma-7b-nlp"]
    chosen, log = None, []
    for c in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(c, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(c, torch_dtype=torch.float32, device_map="cpu")
            chosen = c
            break
        except Exception as e:
            log.append({c: f"{type(e).__name__}: {str(e)[:150]}"})
    if chosen is None:
        return {"status": "error", "tried": candidates, "log": log}
    model.eval()

    def per_pdf(pdf_id, b):
        first_lines = "\n".join(b["full_text"].splitlines()[:8])
        prompt = (f"Analyze the sentiment of this Norwegian financial statement. "
                  f"Provide: negative/neutral/positive.\nText: {first_lines}\nAnswer:")
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ans = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"answer": ans.strip()}

    return {"checkpoint": chosen, "log": log, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("finma", main)
