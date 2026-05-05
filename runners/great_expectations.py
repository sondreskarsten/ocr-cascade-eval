from shared import for_each_pdf, run_with_metrics


def main():
    import pandas as pd
    try:
        import great_expectations as ge
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        rows = [{"page_n": int(k), "n_chars": len(v)} for k, v in b["page_text"].items()]
        df = pd.DataFrame(rows)
        gdf = ge.from_pandas(df)
        results = []
        try:
            r = gdf.expect_column_to_exist("n_chars"); results.append({"check": "col exists", "ok": r["success"]})
            r = gdf.expect_column_values_to_be_between("n_chars", 0, 10000); results.append({"check": "range", "ok": r["success"]})
            r = gdf.expect_column_values_to_not_be_null("page_n"); results.append({"check": "no null", "ok": r["success"]})
        except Exception as e:
            results.append({"error": f"{type(e).__name__}: {e}"})
        return {"n_rows": len(df), "checks": results}

    return {"library": "great_expectations", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("great_expectations", main)
