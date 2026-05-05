from shared import fetch_fixture, run_with_metrics


def _norm(box, w, h):
    return [int(1000 * box[0] / w), int(1000 * box[1] / h),
            int(1000 * box[2] / w), int(1000 * box[3] / h)]


def main():
    fx = fetch_fixture()
    import json, torch
    from PIL import Image
    from transformers import AutoTokenizer, LiltForTokenClassification

    ckpt = "nielsr/lilt-xlm-roberta-base"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = LiltForTokenClassification.from_pretrained(ckpt, num_labels=7)
    model.eval()

    tess = json.loads(open(fx["tesseract_input.json"]).read())
    pages = {}
    for label, n in [("p02", "02"), ("p06", "06")]:
        img = Image.open(fx[f"pages_{label}.png"]).convert("RGB")
        w, h = img.size
        words = tess[n]["words"]
        boxes = [_norm(b, w, h) for b in tess[n]["boxes"]]
        enc = tok(words, boxes=boxes, is_split_into_words=True,
                  return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            out = model(**enc)
        preds = out.logits.argmax(-1)[0].tolist()
        pages[label] = {
            "n_words_in": len(words),
            "n_tokens_out": len(preds),
            "logits_shape": list(out.logits.shape),
            "sample_pred_ids": preds[:30],
            "license": "MIT (commercial-safe)",
        }
    return {"checkpoint": ckpt, "pages": pages,
            "note": "Pretrained backbone with untrained head; demonstrates language-independent layout encoding"}


if __name__ == "__main__":
    run_with_metrics("lilt", main)
