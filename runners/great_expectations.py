from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import pandas as pd
    info = {"library": "great_expectations"}
    try:
        import great_expectations as gx
        info["version"] = gx.__version__
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        return info
    df = pd.read_csv(fx["table.csv"])
    try:
        import great_expectations.expectations as gxe
        ctx = gx.get_context(mode="ephemeral")
        ds = ctx.data_sources.add_pandas("p")
        a = ds.add_dataframe_asset("a")
        bd = a.add_batch_definition_whole_dataframe("bd")
        batch = bd.get_batch(batch_parameters={"dataframe": df})
        results = batch.validate(gxe.ExpectColumnToExist(column="label"))
        info["validation_demo"] = "ok"
        info["result_success"] = bool(results.success)
    except Exception as e:
        info["validation_error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("great_expectations", main)
