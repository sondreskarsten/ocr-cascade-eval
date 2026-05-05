from shared import for_each_pdf, run_with_metrics


def main():
    import pandas as pd
    info = {"library": "great_expectations"}
    try:
        import great_expectations as gx
        info["version"] = getattr(gx, "__version__", "?")
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        rows = [{"page_n": int(k), "n_chars": len(v)} for k, v in b["page_text"].items()]
        df = pd.DataFrame(rows)

        # Use modern GX API
        results = []
        try:
            ctx = gx.get_context(mode="ephemeral")
            ds = ctx.data_sources.add_pandas("pdf_pages")
            asset = ds.add_dataframe_asset(name="pages")
            batch_def = asset.add_batch_definition_whole_dataframe("all")
            batch = batch_def.get_batch(batch_parameters={"dataframe": df})
            r = batch.validate(gx.expectations.ExpectColumnToExist(column="n_chars"))
            results.append({"check": "col exists", "ok": r.success})
            r = batch.validate(gx.expectations.ExpectColumnValuesToBeBetween(column="n_chars", min_value=0, max_value=10000))
            results.append({"check": "range 0-10000", "ok": r.success})
            r = batch.validate(gx.expectations.ExpectColumnValuesToNotBeNull(column="page_n"))
            results.append({"check": "no null", "ok": r.success})
        except Exception as e:
            results.append({"error": f"{type(e).__name__}: {str(e)[:300]}"})
        return {"n_rows": len(df), "checks": results}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("great_expectations", main)
