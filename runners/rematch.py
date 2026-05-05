from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    info = {"library": "rematch", "paper": "ReMatch: Retrieval-Enhanced Schema Matching"}
    try:
        import rematch
        info["version"] = getattr(rematch, "__version__", "?")
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        info["note"] = "Reference implementation typically lives in research repo, not PyPI"
    return info


if __name__ == "__main__":
    run_with_metrics("rematch", main)
