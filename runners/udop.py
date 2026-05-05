from shared import for_each_pdf, run_with_metrics


def _norm(box, w, h):
    return [int(1000 * box[0] / w), int(1000 * box[1] / h),
            int(1000 * box[2] / w), int(1000 * box[3] / h)]


def main():
    import torch
    from PIL import Image
    from transformers import UdopProcessor, UdopForConditionalGeneration

    ckpt = "microsoft/udop-large"
    proc = UdopProcessor.from_pretrained(ckpt, apply_ocr=False)
    model = UdopForConditionalGeneration.from_pretrained(ckpt)
    model.eval()
    prompts = [
        "Question answering. What is the company name?",
        "Question answering. What is the year?",
        "Question answering. What is årsresultat?",
    ]

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            n = int(img_path.split("/")[-1].split("-")[1].split(".")[0])
            w, h = b["page_size"][str(n)]
            pw = b["page_words"][str(n)]
            words = pw["words"]
            boxes = [_norm(bb, w, h) for bb in pw["boxes"]]
            if not words:
                pages.append({"page_n": n, "skipped": "empty"}); continue
            page_answers = []
            img = Image.open(img_path).convert("RGB")
            for q in prompts:
                try:
                    inputs = proc(images=img, text=q, text_pair=words, boxes=boxes,
                                   return_tensors="pt", truncation=True, max_length=1024)
                    with torch.no_grad():
                        pred = model.generate(**inputs, max_new_tokens=32)
                    a = proc.batch_decode(pred, skip_special_tokens=True)[0]
                    page_answers.append({"q": q, "a": a})
                except Exception as e:
                    page_answers.append({"q": q, "error": f"{type(e).__name__}: {e}"})
            pages.append({"page_n": n, "n_words": len(words), "answers": page_answers})
        return {"n_pages": len(pages), "pages": pages}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("udop", main)
