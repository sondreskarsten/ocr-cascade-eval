from shared import for_each_pdf, run_with_metrics


def main():
    import subprocess, pdfplumber

    def per_pdf(pdf_id, b):
        pages = []
        with pdfplumber.open(b["pdf_ocr"]) as pdf:
            for i, p in enumerate(pdf.pages):
                txt = p.extract_text() or ""
                pages.append({"page_n": i + 1, "n_chars": len(txt), "text": txt})
        return {"n_pages": len(pages),
                "total_chars": sum(p["n_chars"] for p in pages),
                "pages": pages}

    return {"engine": "ocrmypdf+tesseract-nor (precomputed)", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("ocrmypdf", main)
