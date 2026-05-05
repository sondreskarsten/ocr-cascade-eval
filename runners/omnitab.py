from shared import for_each_pdf, run_with_metrics


def _extract_first_table(pdf_path):
    import camelot
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        for t in tables:
            df = t.df
            if df.shape[0] >= 3 and df.shape[1] >= 2:
                return df.astype(str)
    except Exception:
        pass
    return None


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    ckpt = "neulab/omnitab-large-finetuned-wtq"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt); model.eval()
    questions = ["What is årsresultat?", "Sum kostnader?"]

    def per_pdf(pdf_id, b):
        df = _extract_first_table(b["pdf_ocr"])
        if df is None:
            return {"status": "no_table_extracted"}
        answers = []
        for q in questions:
            try:
                enc = tok(table=df, query=q, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=32)
                a = tok.batch_decode(out, skip_special_tokens=True)[0]
                answers.append({"q": q, "a": a})
            except Exception as e:
                answers.append({"q": q, "error": f"{type(e).__name__}: {e}"})
        return {"table_shape": list(df.shape), "questions_and_answers": answers}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("omnitab", main)
