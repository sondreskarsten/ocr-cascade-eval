from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from PIL import Image
    from transformers import LayoutLMv3Processor, LiltForTokenClassification

    ckpt = "nielsr/lilt-xlm-roberta-base"
    proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LiltForTokenClassification.from_pretrained(ckpt, num_labels=7)
    model.eval()

    def _norm(box, w, h):
        return [int(1000 * box[0] / w), int(1000 * box[1] / h),
                int(1000 * box[2] / w), int(1000 * box[3] / h)]

    def per_pdf(pdf_id, b):
        pages_out = []
        for img_path, words_meta, sz in zip(b["page_imgs"], b["page_words"], b["page_size"]):
            words = words_meta.get("words", [])
            boxes = words_meta.get("boxes", [])
            if not words:
                pages_out.append({"img": img_path, "skip": "no words"})
                continue
            w, h = sz
            n_boxes = [_norm(box, w, h) for box in boxes]
            img = Image.open(img_path).convert("RGB")
            try:
                enc = proc(img, words, boxes=n_boxes, return_tensors="pt",
                           truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"],
                                bbox=enc["bbox"])
                preds = out.logits.argmax(-1)[0].tolist()
                pages_out.append({"n_words_in": len(words),
                                  "n_tokens_out": len(preds),
                                  "logits_shape": list(out.logits.shape),
                                  "sample_pred_ids": preds[:30]})
            except Exception as e:
                pages_out.append({"error": f"{type(e).__name__}: {e}"})
        return {"checkpoint": ckpt, "pages": pages_out,
                "license": "MIT (commercial-safe)",
                "note": "Untrained head; demonstrates language-independent layout encoding"}

    return {"per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("lilt", main)
