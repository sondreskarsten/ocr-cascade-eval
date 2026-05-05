from shared import for_each_pdf, run_with_metrics


def _extract(pdf, mode):
    import tabula
    kwargs = {mode: True}
    dfs = tabula.read_pdf(pdf, pages="all", **kwargs)
    return [{"shape": list(df.shape),
             "columns": [str(c) for c in df.columns.tolist()],
             "first_rows": df.head(5).fillna("").astype(str).values.tolist()}
            for df in dfs]


def main():
    def per_pdf(pdf_id, b):
        out = {}
        for source_label, pdf_key in [("raw_image", "pdf"), ("after_ocrmypdf", "pdf_ocr")]:
            out[source_label] = {}
            for mode in ("lattice", "stream"):
                try:
                    tabs = _extract(b[pdf_key], mode)
                    out[source_label][mode] = {"n_tables": len(tabs), "tables": tabs}
                except Exception as e:
                    out[source_label][mode] = {"error": f"{type(e).__name__}: {e}"}
        return out

    return {"per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("tabula", main)
