from shared import fetch_fixture, run_with_metrics


def _normalize_box(box, w, h):
    return [int(1000 * box[0] / w), int(1000 * box[1] / h),
            int(1000 * box[2] / w), int(1000 * box[3] / h)]


def main():
    fx = fetch_fixture()
    import json, torch
    from PIL import Image
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

    ckpt = "microsoft/layoutlmv3-base"
    proc = LayoutLMv3Processor.from_pretrained(ckpt, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(ckpt, num_labels=7)
    model.eval()

    tess = json.loads(open(fx["tesseract_input.json"]).read())
    pages = {}
    for label, n in [("p02", "02"), ("p06", "06")]:
        img = Image.open(fx[f"pages_{label}.png"]).convert("RGB")
        w, h = img.size
        words = tess[n]["words"]
        boxes = [_normalize_box(b, w, h) for b in tess[n]["boxes"]]
        inputs = proc(img, words, boxes=boxes, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**inputs)
        preds = out.logits.argmax(-1)[0].tolist()
        pages[label] = {
            "n_words_in": len(words),
            "n_tokens_out": len(preds),
            "logits_shape": list(out.logits.shape),
            "sample_pred_ids": preds[:30],
            "license": "CC-BY-NC (non-commercial only)",
        }
    return {"checkpoint": ckpt, "pages": pages,
            "note": "Pretrained head untrained -> labels uncalibrated; demonstrates load+inference path"}


if __name__ == "__main__":
    run_with_metrics("layoutlmv3", main)
