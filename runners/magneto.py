from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, subprocess, os
    samples = json.loads(open(fx["samples.json"]).read())
    info = {"library": "magneto", "paper": "arXiv:2412.08194"}
    try:
        subprocess.run(["pip", "install", "magneto-python"],
                       capture_output=True, timeout=120)
        import magneto
        info["version"] = getattr(magneto, "__version__", "?")
        info["module"] = sorted([x for x in dir(magneto) if not x.startswith("_")])[:30]
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["fallback"] = "Magneto is not on PyPI; try: pip install git+https://github.com/dataneuro/magneto"
    return info


if __name__ == "__main__":
    run_with_metrics("magneto", main)
