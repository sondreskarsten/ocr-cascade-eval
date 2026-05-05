from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())
    canonical = samples["canonical_titles"][:20]

    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    ckpt = "knowledgator/gliclass-large-v1.0"
    model = GLiClassModel.from_pretrained(ckpt)
    if not hasattr(model, "_supports_sdpa"):
        model._supports_sdpa = False
    tok = AutoTokenizer.from_pretrained(ckpt)
    pipe = ZeroShotClassificationPipeline(model, tok, classification_type="multi-label", device="cpu")

    def per_pdf(pdf_id, b):
        # take the note titles (heuristic: short capitalized lines)
        seen = []
        for ln in b["full_text"].splitlines():
            ln = ln.strip()
            if 4 <= len(ln) <= 60 and not ln.replace(" ","").isdigit() and ln[0].isupper():
                if ln not in seen: seen.append(ln)
            if len(seen) >= 15: break
        results = []
        for ln in seen:
            try:
                r = pipe(ln, canonical, threshold=0.0)
                top3 = sorted(r, key=lambda x: -x.get("score",0))[:3] if isinstance(r, list) else r
                results.append({"line": ln, "top3": top3})
            except Exception as e:
                results.append({"line": ln, "error": f"{type(e).__name__}: {e}"})
        return {"n_inputs": len(seen), "results": results}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("gliclass", main)
