from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())
    import torch
    from PIL import Image
    from colpali_engine.models import ColPali, ColPaliProcessor
    ckpt = "vidore/colpali-v1.3"
    proc = ColPaliProcessor.from_pretrained(ckpt)
    model = ColPali.from_pretrained(ckpt, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    images = [Image.open(fx[f"pages_p{n}.png"]).convert("RGB") for n in ("02","06")]
    queries = [item["q"] for item in samples["queries"][:5]]
    with torch.no_grad():
        img_inputs = proc.process_images(images)
        img_emb = model(**img_inputs)
        q_inputs = proc.process_queries(queries)
        q_emb = model(**q_inputs)
        scores = proc.score_multi_vector(q_emb, img_emb)
    return {"checkpoint": ckpt,
            "n_pages": len(images), "n_queries": len(queries),
            "scores": scores.tolist()}


if __name__ == "__main__":
    run_with_metrics("colpali", main)
