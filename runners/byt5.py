from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    ckpt = "google/byt5-small"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt); model.eval()

    def per_pdf(pdf_id, b):
        lines = []
        for ln in b["full_text"].splitlines():
            ln = ln.strip()
            if 5 <= len(ln) <= 60 and ln not in lines: lines.append(ln)
            if len(lines) >= 8: break
        outputs = []
        for ln in lines:
            inputs = tok([ln], return_tensors="pt", padding=True)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64)
            text = tok.batch_decode(out, skip_special_tokens=True)[0]
            outputs.append({"input": ln, "output": text})
        return {"n_inputs": len(lines), "outputs": outputs,
                "note": "byt5-small base, no task fine-tuning — output reflects vanilla LM behaviour"}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("byt5", main)
