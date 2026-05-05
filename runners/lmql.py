from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    info = {"library": "lmql"}
    try:
        import lmql
        info["version"] = getattr(lmql, "__version__", "?")
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
    info["note"] = "LMQL is a query language; needs an LM backend"
    return info


if __name__ == "__main__":
    run_with_metrics("lmql", main)
