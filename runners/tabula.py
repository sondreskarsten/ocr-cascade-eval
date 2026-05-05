from shared import fetch_fixture, run_with_metrics


def _extract(pdf, mode):
    import tabula
    kwargs = {mode: True}
    dfs = tabula.read_pdf(pdf, pages="2,6", **kwargs)
    return [
        {"shape": list(df.shape),
         "columns": [str(c) for c in df.columns.tolist()],
         "first_rows": df.head(5).fillna("").astype(str).values.tolist()}
        for df in dfs
    ]


def main():
    fx = fetch_fixture()
    out = {}
    for label, key in [("raw_image", "test.pdf"), ("after_ocrmypdf", "test_ocr.pdf")]:
        out[label] = {}
        for mode in ("lattice", "stream"):
            try:
                tables = _extract(fx[key], mode)
                out[label][mode] = {"n_tables": len(tables), "tables": tables}
            except Exception as e:
                out[label][mode] = {"error": f"{type(e).__name__}: {e}"}
    return {"results": out}


if __name__ == "__main__":
    run_with_metrics("tabula", main)
