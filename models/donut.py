from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import torch
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    ckpt = "naver-clova-ix/donut-base-finetuned-cord-v2"
    proc = DonutProcessor.from_pretrained(ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt)
    model.eval()
    task = "<s_cord-v2>"
    decoder_in = proc.tokenizer(task, add_special_tokens=False, return_tensors="pt").input_ids

    pages = {}
    for label, key in [("p02", "pages_p02.png"), ("p06", "pages_p06.png")]:
        img = Image.open(fx[key]).convert("RGB").resize((1280, 960))
        pixel_values = proc(img, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = model.generate(
                pixel_values, decoder_input_ids=decoder_in,
                max_length=1024, pad_token_id=proc.tokenizer.pad_token_id,
                eos_token_id=proc.tokenizer.eos_token_id,
                use_cache=True, num_beams=1,
            )
        seq = proc.batch_decode(outputs)[0]
        seq = seq.replace(proc.tokenizer.eos_token, "").replace(proc.tokenizer.pad_token, "")
        try:
            parsed = proc.token2json(seq)
        except Exception as e:
            parsed = {"_parse_error": str(e)}
        pages[label] = {"raw_seq": seq[:2000], "parsed": parsed}
    return {"checkpoint": ckpt, "pages": pages}


if __name__ == "__main__":
    run_with_metrics("donut", main)
