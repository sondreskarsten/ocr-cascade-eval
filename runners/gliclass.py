from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    samples = json.loads(open(fx["samples.json"]).read())
    ckpt = "knowledgator/gliclass-large-v1.0"
    model = GLiClassModel.from_pretrained(ckpt)
    tok = AutoTokenizer.from_pretrained(ckpt)
    pipe = ZeroShotClassificationPipeline(model, tok, classification_type="multi-label", device="cpu")

    label = samples["norwegian_label"]
    candidates = samples["candidates"]
    res = pipe(label, candidates, threshold=0.0)
    return {"checkpoint": ckpt, "input": label, "candidates": candidates, "scores": res}


if __name__ == "__main__":
    run_with_metrics("gliclass", main)
