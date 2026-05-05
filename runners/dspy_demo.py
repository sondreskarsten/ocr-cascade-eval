from shared import for_each_pdf, run_with_metrics


def main():
    info = {"library": "dspy"}
    try:
        import dspy
        info["version"] = getattr(dspy, "__version__", "?")
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}"}

    class ExtractFinancials(dspy.Signature):
        """Extract company name and årsresultat from Norwegian regnskap text."""
        text: str = dspy.InputField()
        company_name: str = dspy.OutputField()
        arsresultat: str = dspy.OutputField()

    def per_pdf(pdf_id, b):
        return {"signature": "ExtractFinancials(text -> company_name, arsresultat)",
                "input_excerpt": b["full_text"][:300],
                "input_chars": len(b["full_text"]),
                "note": "DSPy needs an LM provider to actually run; smoke test demonstrates signature compilation"}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("dspy", main)
