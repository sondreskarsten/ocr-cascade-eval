from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    info = {"library": "dedupe"}
    try:
        import dedupe
        info["version"] = getattr(dedupe, "__version__", "?")
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
    info["note"] = "dedupe is supervised entity resolution; needs labeled training pairs."
    return info


if __name__ == "__main__":
    run_with_metrics("dedupe", main)
