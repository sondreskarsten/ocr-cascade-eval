from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    ckpt = "microsoft/trocr-base-printed"
    proc = TrOCRProcessor.from_pretrained(ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt)
    model.eval()
    pages = {}
    for label, key in [("p02", "pages_p02.png"), ("p06", "pages_p06.png")]:
        img = Image.open(fx[key]).convert("RGB")
        crop = img.crop((0, 0, min(img.width, 800), min(img.height, 100)))
        pixel = proc(images=crop, return_tensors="pt").pixel_values
        with torch.no_grad():
            out = model.generate(pixel, max_new_tokens=128)
        text = proc.batch_decode(out, skip_special_tokens=True)[0]
        pages[label] = {"text_strip": text,
                        "note": "TrOCR is line-level only; full-page OCR needs detector first"}
    return {"checkpoint": ckpt, "pages": pages}


if __name__ == "__main__":
    run_with_metrics("trocr", main)
