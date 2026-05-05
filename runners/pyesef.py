from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import urllib.request, zipfile, os, glob

    url = "https://filings.xbrl.org/216-100-0006-50-2022-12-31-ESEF-NO-0/reports/216-100-0006-50-2022-12-31-en.zip"
    z = "/tmp/esef.zip"
    d = "/tmp/esef"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(z, "wb") as f:
            f.write(r.read())
        os.makedirs(d, exist_ok=True)
        zipfile.ZipFile(z).extractall(d)
    except Exception as e:
        return {"status": "error", "fetch_error": f"{type(e).__name__}: {e}"}

    try:
        import pyesef
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}",
                "note": "pyesef may not be on PyPI; check repo at https://github.com/jeppe-djuhrsen/pyesef"}

    files = glob.glob(f"{d}/**/*.xhtml", recursive=True) + glob.glob(f"{d}/**/*.html", recursive=True)
    sample = files[0] if files else None
    info = {"pyesef_version": getattr(pyesef, "__version__", "?"),
            "report_file": sample,
            "module_dir": dir(pyesef)[:30]}
    if hasattr(pyesef, "extract") and sample:
        try:
            facts = pyesef.extract(sample)
            info["sample_n_facts"] = len(facts) if hasattr(facts, "__len__") else None
        except Exception as e:
            info["extract_error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("pyesef", main)
