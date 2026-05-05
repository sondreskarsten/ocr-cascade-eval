from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())
    canonical = samples["canonical_titles"][:30]

    info = {"library": "outlines"}
    try:
        import outlines
        info["version"] = getattr(outlines, "__version__", "?")
        info["api"] = sorted([a for a in dir(outlines) if not a.startswith("_")])[:60]
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}"}

    smoke = {}
    try:
        if hasattr(outlines, "models") and hasattr(outlines.models, "transformers"):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            base = "Qwen/Qwen2.5-0.5B"
            tok = AutoTokenizer.from_pretrained(base)
            mdl = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")
            m = outlines.from_transformers(mdl, tok) if hasattr(outlines, "from_transformers") \
                else outlines.models.transformers(base)
            from outlines.types import Choice
            choice_type = Choice(canonical[:10])
            prompt = f"Map this Norwegian financial label to canonical:\n{samples['norwegian_label']}\nCanonical:"
            ans = m(prompt, choice_type, max_new_tokens=64)
            smoke = {"prompt": prompt, "answer": str(ans), "choices": canonical[:10]}
        else:
            smoke = {"note": "outlines.models.transformers not present in this version"}
    except Exception as e:
        smoke = {"smoke_test_error": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        return {"input_chars": len(b["full_text"]),
                "input_excerpt": b["full_text"][:300],
                "constrained_to": canonical[:5],
                "note": "constrained generation requires LM backend; smoke test in info"}

    return {**info, "smoke_test": smoke, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("outlines", main)
