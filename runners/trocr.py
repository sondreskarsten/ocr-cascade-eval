from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    ckpt = "microsoft/trocr-base-printed"
    proc = TrOCRProcessor.from_pretrained(ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt)
    model.eval()

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            img = Image.open(img_path).convert("RGB")
            pixel_values = proc(img, return_tensors="pt").pixel_values
            with torch.no_grad():
                out = model.generate(pixel_values, max_new_tokens=64)
            text = proc.batch_decode(out, skip_special_tokens=True)[0]
            pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                          "n_chars": len(text), "text": text})
        return {"n_pages": len(pages),
                "total_chars": sum(p["n_chars"] for p in pages),
                "pages": pages,
                "note": "TrOCR processes whole image as one line — designed for line-level not page OCR; output is degraded for full-page input"}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("trocr", main)
