from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import torch
    from PIL import Image
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    ckpt = "mistralai/Pixtral-12B-2409"
    info = {"checkpoint": ckpt, "size_gb_estimate": 24}
    try:
        proc = AutoProcessor.from_pretrained(ckpt)
        model = LlavaForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.float32)
        model.eval()
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["note"] = "Pixtral 12B requires 24GB+ RAM and HF auth; likely OOM on standard Cloud Run"
        return info
    img = Image.open(fx["pages_p02.png"]).convert("RGB")
    prompt = "[IMG]Extract all text from this Norwegian financial report page."
    inputs = proc(text=prompt, images=img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512)
    info["completion"] = proc.batch_decode(out, skip_special_tokens=True)[0][:2000]
    return info


if __name__ == "__main__":
    run_with_metrics("pixtral", main)
