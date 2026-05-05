from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from PIL import Image
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

    ckpt = "google/pix2struct-docvqa-base"
    proc = Pix2StructProcessor.from_pretrained(ckpt)
    model = Pix2StructForConditionalGeneration.from_pretrained(ckpt)
    model.eval()

    # Financial questions targeting nøkkeltall
    questions = [
        "What is the company name?",
        "What is the year?",
        "What is the årsresultat?",
        "What is the sum eiendeler?",
        "What is the driftsresultat?",
    ]

    def per_pdf(pdf_id, b):
        # Run questions on page 2 only (resultatregnskap typically) to fit timeout budget
        n = len(b["page_imgs"])
        if n >= 2:
            target_imgs = [b["page_imgs"][1]]
        elif n >= 1:
            target_imgs = [b["page_imgs"][0]]
        else:
            return {"checkpoint": ckpt, "error": "no pages"}
        page_results = []
        for img_path in target_imgs:
            img = Image.open(img_path).convert("RGB")
            qa = []
            for q in questions:
                try:
                    inputs = proc(images=img, return_tensors="pt", text=q)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=48)
                    a = proc.decode(out[0], skip_special_tokens=True)
                    qa.append({"q": q, "a": a})
                except Exception as e:
                    qa.append({"q": q, "error": f"{type(e).__name__}: {e}"})
            page_results.append({"img": img_path.split("/")[-1], "qa": qa})
        return {"checkpoint": ckpt, "task": "DocVQA per-page",
                "n_pages_queried": len(page_results), "pages": page_results}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("pix2struct", main)
