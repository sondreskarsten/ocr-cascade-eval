from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import torch
    from PIL import Image
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    ckpt = "google/pix2struct-docvqa-base"
    proc = Pix2StructProcessor.from_pretrained(ckpt)
    model = Pix2StructForConditionalGeneration.from_pretrained(ckpt)
    model.eval()
    img = Image.open(fx["pages_p02.png"]).convert("RGB").resize((1024, 1024))
    questions = ["What is the company name?", "What is årsresultat 2024?"]
    answers = []
    for q in questions:
        inputs = proc(images=img, text=q, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        answers.append({"q": q, "a": proc.decode(out[0], skip_special_tokens=True)})
    return {"checkpoint": ckpt, "qa": answers}


if __name__ == "__main__":
    run_with_metrics("pix2struct", main)
