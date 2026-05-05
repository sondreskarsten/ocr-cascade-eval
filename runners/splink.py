from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, pandas as pd
    samples = json.loads(open(fx["samples.json"]).read())
    info = {"library": "splink"}
    try:
        import splink
        info["version"] = getattr(splink, "__version__", "?")
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        return info
    info["n_canonical"] = len(samples["canonical_titles"])
    info["note"] = "Splink targets multi-attribute record linkage; not a direct fit for short label mapping."
    return info


if __name__ == "__main__":
    run_with_metrics("splink", main)
