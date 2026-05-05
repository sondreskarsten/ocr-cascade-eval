from shared import for_each_pdf, run_with_metrics


def main():
    import torch
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    ckpt = "naver-clova-ix/donut-base-finetuned-cord-v2"
    proc = DonutProcessor.from_pretrained(ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt)
    model.eval()
    decoder_in = proc.tokenizer("<s_cord-v2>", add_special_tokens=False, return_tensors="pt").input_ids

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            img = Image.open(img_path).convert("RGB").resize((1280, 960))
            pixel_values = proc(img, return_tensors="pt").pixel_values
            try:
                with torch.no_grad():
                    out = model.generate(
                        pixel_values, decoder_input_ids=decoder_in,
                        max_length=512, pad_token_id=proc.tokenizer.pad_token_id,
                        eos_token_id=proc.tokenizer.eos_token_id,
                        use_cache=True, num_beams=1,
                    )
                seq = proc.batch_decode(out)[0]
                seq = seq.replace(proc.tokenizer.eos_token, "").replace(proc.tokenizer.pad_token, "")
                try:
                    parsed = proc.token2json(seq)
                except Exception:
                    parsed = {"_raw": seq[:1500]}
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "raw": seq[:800], "parsed": parsed})
            except Exception as e:
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "error": f"{type(e).__name__}: {e}"})
        return {"n_pages": len(pages), "pages": pages}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("donut", main)
