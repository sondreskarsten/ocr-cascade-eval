from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from PIL import Image
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

    ckpt = "google/pix2struct-base"
    proc = Pix2StructProcessor.from_pretrained(ckpt)
    model = Pix2StructForConditionalGeneration.from_pretrained(ckpt)
    model.eval()

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            img = Image.open(img_path).convert("RGB")
            inputs = proc(images=img, return_tensors="pt", text="What is the main heading?")
            try:
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=64)
                text = proc.decode(out[0], skip_special_tokens=True)
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "answer": text})
            except Exception as e:
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "error": f"{type(e).__name__}: {e}"})
        return {"n_pages": len(pages), "pages": pages,
                "task": "VQA: 'What is the main heading?' per page"}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("pix2struct", main)
