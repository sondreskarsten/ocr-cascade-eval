from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    ckpt = "ltg/nort5-base"
    try:
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)
    except Exception as e1:
        try:
            tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        except Exception as e2:
            return {"status": "error", "checkpoint": ckpt,
                    "tokenizer_errors": [f"slow: {e1}", f"fast: {e2}"]}
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, trust_remote_code=True)
        model.eval()
    except Exception as e:
        return {"status": "error", "checkpoint": ckpt,
                "model_error": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        lines = []
        for ln in b["full_text"].splitlines():
            ln = ln.strip()
            if 5 <= len(ln) <= 80 and ln not in lines: lines.append(ln)
            if len(lines) >= 5: break
        outs = []
        for ln in lines:
            try:
                inputs = tok(ln, return_tensors="pt")
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=32)
                text = tok.decode(out[0], skip_special_tokens=True)
                outs.append({"input": ln, "output": text})
            except Exception as e:
                outs.append({"input": ln, "error": f"{type(e).__name__}: {e}"})
        return {"outputs": outs}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("nort5", main)
