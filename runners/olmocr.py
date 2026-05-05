from shared import for_each_pdf, run_with_metrics


def main():
    info = {"library": "olmocr"}
    try:
        import olmocr
        info["version"] = getattr(olmocr, "__version__", "?")
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}",
                "note": "olmocr needs vision LM backend; package may not be installed"}

    def per_pdf(pdf_id, b):
        return {"n_pages": b["n_pages"], "pdf": b["pdf"],
                "note": "olmocr needs Qwen2-VL backend (>4B params); skipped on CPU"}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("olmocr", main)
