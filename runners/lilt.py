from shared import for_each_pdf, run_with_metrics


def _norm(box, w, h):
    return [int(1000 * box[0] / w), int(1000 * box[1] / h),
            int(1000 * box[2] / w), int(1000 * box[3] / h)]


def main():
    import torch
    from PIL import Image
    from transformers import AutoTokenizer, LayoutLMv3TokenizerFast, LiltForTokenClassification

    ckpt = "nielsr/lilt-xlm-roberta-base"
    try:
        from transformers import LayoutXLMTokenizerFast
        tok = LayoutXLMTokenizerFast.from_pretrained("microsoft/layoutxlm-base")
    except Exception:
        tok = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    model = LiltForTokenClassification.from_pretrained(ckpt, num_labels=7)
    model.eval()

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            n = int(img_path.split("/")[-1].split("-")[1].split(".")[0])
            w, h = b["page_size"][str(n)]
            page_words = b["page_words"][str(n)]
            words = page_words["words"]
            boxes = [_norm(bb, w, h) for bb in page_words["boxes"]]
            if not words:
                pages.append({"page_n": n, "skipped": "empty page"}); continue
            try:
                enc = tok(words, boxes=boxes, is_split_into_words=True,
                          return_tensors="pt", truncation=True, max_length=512, padding=True)
                enc.pop("pixel_values", None)
                with torch.no_grad():
                    out = model(**enc)
                preds = out.logits.argmax(-1)[0].tolist()
                pages.append({"page_n": n, "n_words_in": len(words),
                              "logits_shape": list(out.logits.shape),
                              "sample_preds": preds[:30]})
            except Exception as e:
                pages.append({"page_n": n, "error": f"{type(e).__name__}: {e}"})
        return {"n_pages": len(pages), "pages": pages,
                "license": "MIT", "tokenizer": str(type(tok).__name__)}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("lilt", main)
