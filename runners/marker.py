from shared import for_each_pdf, run_with_metrics


def main():
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}"}
    converter = PdfConverter(artifact_dict=create_model_dict())

    def per_pdf(pdf_id, b):
        try:
            r = converter(b["pdf"])
            md = r.markdown if hasattr(r, "markdown") else str(r)
            return {"n_chars_markdown": len(md), "markdown_preview": md[:1500]}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    return {"library": "marker-pdf", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("marker", main)
