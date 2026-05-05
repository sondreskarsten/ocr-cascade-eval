from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())
    info = {"library": "guidance"}
    try:
        import guidance
        info["version"] = getattr(guidance, "__version__", "?")
        info["module"] = [x for x in dir(guidance) if not x.startswith("_")][:30]
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        return info
    info["constraint_demo_n_choices"] = len(samples["canonical_titles"][:30])
    return info


if __name__ == "__main__":
    run_with_metrics("guidance", main)
