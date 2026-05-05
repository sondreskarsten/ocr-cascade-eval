from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    info = {"library": "schemora", "paper": "EMNLP 2025 hybrid framework"}
    try:
        import schemora
        info["version"] = getattr(schemora, "__version__", "?")
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        info["note"] = "Reference implementation typically not on PyPI"
    return info


if __name__ == "__main__":
    run_with_metrics("schemora", main)
