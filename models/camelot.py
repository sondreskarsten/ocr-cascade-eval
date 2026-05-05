from shared import fetch_fixture, run_with_metrics


def _extract(pdf, flavor):
    import camelot
    tables = camelot.read_pdf(pdf, pages="2,6", flavor=flavor)
    return [
        {"page": t.page, "shape": list(t.shape),
         "accuracy": round(t.parsing_report.get("accuracy", 0), 2),
         "first_rows": t.df.head(5).values.tolist()}
        for t in tables
    ]


def main():
    fx = fetch_fixture()
    out = {}
    for label, key in [("raw_image", "test.pdf"), ("after_ocrmypdf", "test_ocr.pdf")]:
        out[label] = {}
        for flavor in ("lattice", "stream"):
            try:
                tables = _extract(fx[key], flavor)
                out[label][flavor] = {"n_tables": len(tables), "tables": tables}
            except Exception as e:
                out[label][flavor] = {"error": f"{type(e).__name__}: {e}"}
    return {"results": out}


if __name__ == "__main__":
    run_with_metrics("camelot", main)
