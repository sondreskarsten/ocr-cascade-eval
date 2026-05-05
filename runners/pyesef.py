from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import urllib.request, os

    info = {}
    try:
        import pyesef
        info["pyesef_version"] = getattr(pyesef, "__version__", "?")
        info["pyesef_api"] = [x for x in dir(pyesef) if not x.startswith("_")][:30]
    except Exception as e:
        return {"status": "import_only_failed",
                "import_error": f"{type(e).__name__}: {e}",
                "note": "pyesef not on PyPI in this version; arelle is canonical"}

    sample_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    target = "/tmp/aapl.htm"
    try:
        req = urllib.request.Request(sample_url,
            headers={"User-Agent": "Sondre Skarsten sondre@example.no"})
        with urllib.request.urlopen(req, timeout=120) as r, open(target, "wb") as f:
            f.write(r.read())
        info["fetched_bytes"] = os.path.getsize(target)
        info["sample_url"] = sample_url
    except Exception as e:
        info["fetch_error"] = f"{type(e).__name__}: {e}"
        return info

    if hasattr(pyesef, "extract"):
        try:
            facts = pyesef.extract(target)
            info["sample_n_facts"] = len(facts) if hasattr(facts, "__len__") else None
        except Exception as e:
            info["extract_error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("pyesef", main)
