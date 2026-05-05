from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from PIL import Image

    try:
        from transformers import NougatProcessor, VisionEncoderDecoderModel
    except Exception as e:
        return {"status": "error", "import_error": str(e)}

    ckpt = "facebook/nougat-small"
    proc = NougatProcessor.from_pretrained(ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt)
    model.eval()

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            img = Image.open(img_path).convert("RGB")
            pixel_values = proc(img, return_tensors="pt").pixel_values
            try:
                with torch.no_grad():
                    out = model.generate(pixel_values, min_length=1, max_new_tokens=512,
                                          bad_words_ids=[[proc.tokenizer.unk_token_id]])
                seq = proc.batch_decode(out, skip_special_tokens=True)[0]
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "n_chars": len(seq), "text": seq[:1500]})
            except Exception as e:
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "error": f"{type(e).__name__}: {e}"})
        return {"n_pages": len(pages),
                "total_chars": sum(p.get("n_chars", 0) for p in pages),
                "pages": pages}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("nougat", main)
