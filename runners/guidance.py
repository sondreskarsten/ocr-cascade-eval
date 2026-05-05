from shared import for_each_pdf, run_with_metrics


def main():
    info = {"library": "guidance"}
    try:
        import guidance as M
        info["version"] = getattr(M, "__version__", "?")
        info["module_dir"] = sorted([a for a in dir(M) if not a.startswith("_")])[:40]
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        return {"input_chars": len(b["full_text"]),
                "first_line": b["full_text"].splitlines()[0][:200] if b["full_text"] else "",
                "note": "guidance smoke test — module imported, full task needs LM/backend setup"}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("guidance", main)
