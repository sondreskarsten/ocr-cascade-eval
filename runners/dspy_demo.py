from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())
    info = {"library": "dspy"}
    try:
        import dspy
        info["version"] = getattr(dspy, "__version__", "?")
        info["module_top_level"] = sorted([x for x in dir(dspy) if not x.startswith("_")])[:40]
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        return info
    info["smoke_test"] = {
        "available_optimizers": [n for n in ["MIPROv2", "GEPA", "SIMBA", "BootstrapFewShot",
                                              "BootstrapFewShotWithRandomSearch", "COPRO"]
                                  if hasattr(dspy, n) or hasattr(getattr(dspy, "teleprompt", None), n)],
        "n_canonical_choices": len(samples["canonical_titles"]),
        "note": "DSPy needs an LM provider. No API key in this run."
    }
    return info


if __name__ == "__main__":
    run_with_metrics("dspy", main)
