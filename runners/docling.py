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
            if len(md) <= 50000:
                # Store full markdown — small enough to audit fully
                return {"n_chars_markdown": len(md), "markdown": md}
            else:
                # Large doc — store head/mid/tail samples
                return {"n_chars_markdown": len(md),
                        "preview_head": md[:1500],
                        "preview_mid": md[len(md)//2:len(md)//2+1500],
                        "preview_tail": md[-1500:],
                        "truncated": True}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    return {"library": "docling", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("docling", main)
