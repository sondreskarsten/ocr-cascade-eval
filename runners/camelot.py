from shared import for_each_pdf, run_with_metrics


def _extract(pdf, pages_arg, flavor):
    import camelot
    tables = camelot.read_pdf(pdf, pages=pages_arg, flavor=flavor)
    return [{"page": t.page, "shape": list(t.shape),
             "accuracy": round(t.parsing_report.get("accuracy", 0), 2),
             "first_rows": t.df.head(5).values.tolist()} for t in tables]


def main():
    def per_pdf(pdf_id, b):
        out = {}
        for source_label, pdf_key in [("raw_image", "pdf"), ("after_ocrmypdf", "pdf_ocr")]:
            out[source_label] = {}
            for flavor in ("lattice", "stream"):
                try:
                    tables = _extract(b[pdf_key], "all", flavor)
                    out[source_label][flavor] = {"n_tables": len(tables), "tables": tables}
                except Exception as e:
                    out[source_label][flavor] = {"error": f"{type(e).__name__}: {e}"}
        return out

    return {"per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("camelot", main)
