from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ckpt = "ChanceFocus/finma-7b-nlp"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    def per_pdf(pdf_id, b):
        excerpt = b["full_text"][:1500]
        prompt = ("Analyze the sentiment of this financial text. Provide answer as negative, positive, or neutral.\n"
                  f"Text: {excerpt}\nAnswer:")
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1500)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ans = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"input_chars": len(excerpt), "answer": ans.strip()}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("finma", main)
