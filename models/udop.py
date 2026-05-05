from shared import fetch_fixture, run_with_metrics


def _norm(box, w, h):
    return [int(1000 * box[0] / w), int(1000 * box[1] / h),
            int(1000 * box[2] / w), int(1000 * box[3] / h)]


def main():
    fx = fetch_fixture()
    import json, torch
    from PIL import Image
    from transformers import UdopProcessor, UdopForConditionalGeneration

    ckpt = "microsoft/udop-large"
    proc = UdopProcessor.from_pretrained(ckpt, apply_ocr=False)
    model = UdopForConditionalGeneration.from_pretrained(ckpt)
    model.eval()

    tess = json.loads(open(fx["tesseract_input.json"]).read())
    pages = {}
    prompts = ["Question answering. What is the company name?",
               "Question answering. What is the total revenue 2024?"]
    for label, n in [("p02", "02"), ("p06", "06")]:
        img = Image.open(fx[f"pages_{label}.png"]).convert("RGB")
        w, h = img.size
        words = tess[n]["words"]
        boxes = [_norm(b, w, h) for b in tess[n]["boxes"]]
        page_answers = []
        for q in prompts:
            inputs = proc(images=img, text=q, text_pair=words, boxes=boxes,
                          return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                pred = model.generate(**inputs, max_new_tokens=64)
            ans = proc.batch_decode(pred, skip_special_tokens=True)[0]
            page_answers.append({"q": q, "a": ans})
        pages[label] = {"n_words_in": len(words), "answers": page_answers}
    return {"checkpoint": ckpt, "pages": pages}


if __name__ == "__main__":
    run_with_metrics("udop", main)
