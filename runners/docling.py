from shared import for_each_pdf, run_with_metrics


def main():
    try:
        from docling.document_converter import DocumentConverter
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}"}
    conv = DocumentConverter()

    def per_pdf(pdf_id, b):
        try:
            r = conv.convert(b["pdf"])
            md = r.document.export_to_markdown()
            return {"n_chars_markdown": len(md), "markdown_preview": md[:1500]}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    return {"library": "docling", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("docling", main)
