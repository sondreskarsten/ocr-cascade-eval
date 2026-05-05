from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())

    info = {"library": "marvin"}
    try:
        import marvin
        info["version"] = getattr(marvin, "__version__", "?")
        info["module_top_level"] = sorted([x for x in dir(marvin) if not x.startswith("_")])[:40]
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        return info

    info["smoke_test"] = {
        "input_label": samples["norwegian_label"],
        "n_canonical_choices": len(samples["canonical_titles"]),
        "note": "marvin wraps an LLM client (OpenAI). No API key in this run; smoke test only.",
    }
    return info


if __name__ == "__main__":
    run_with_metrics("marvin", main)
