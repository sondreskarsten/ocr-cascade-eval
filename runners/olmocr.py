from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    info = {"engine": "olmocr", "checkpoint_target": "allenai/olmOCR-7B-0825"}
    try:
        import olmocr
        info["version"] = getattr(olmocr, "__version__", "?")
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        info["note"] = "olmOCR ships as a pipeline package"
    return info


if __name__ == "__main__":
    run_with_metrics("olmocr", main)
