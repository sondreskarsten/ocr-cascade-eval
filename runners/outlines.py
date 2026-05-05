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
        info["api"] = sorted([a for a in dir(outlines) if not a.startswith("_")])
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        excerpt = b["full_text"][:1500]
        return {"input_chars": len(excerpt),
                "input_excerpt": excerpt[:300],
                "constrained_to": canonical[:5],
                "note": "outlines needs an LM backend to actually constrain. Smoke test only."}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("outlines", main)
