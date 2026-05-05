from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import urllib.request, zipfile, os, glob

    candidates = [
        "https://filings.xbrl.org/213800FF8F9OG6MTUE31/2024-12-31/ESEF/NO/0/213800FF8F9OG6MTUE31-2024-12-31-ESEF-NO-0.zip",
        "https://filings.xbrl.org/549300SUWCZWERMVB019/2024-12-31/ESEF/NO/0/549300SUWCZWERMVB019-2024-12-31-ESEF-NO-0.zip",
        "https://filings.xbrl.org/2138006O0X73VFNUH294/2023-12-31/ESEF/NO/0/2138006O0X73VFNUH294-2023-12-31-ESEF-NO-0.zip",
    ]
    fetch_log = []
    z, d = "/tmp/esef.zip", "/tmp/esef"
    fetched_url = None
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(z, "wb") as f:
                f.write(r.read())
            os.makedirs(d, exist_ok=True)
            zipfile.ZipFile(z).extractall(d)
            fetched_url = url
            break
        except Exception as e:
            fetch_log.append({url: f"{type(e).__name__}: {e}"})

    if fetched_url is None:
        return {"status": "error", "fetch_log": fetch_log,
                "note": "all candidate ESEF URLs returned errors"}

    try:
        import pyesef
    except Exception as e:
        return {"status": "import_only_failed", "fetched_url": fetched_url,
                "import_error": f"{type(e).__name__}: {e}",
                "note": "pyesef may not be on PyPI; arelle is canonical"}

    files = glob.glob(f"{d}/**/*.xhtml", recursive=True) + glob.glob(f"{d}/**/*.html", recursive=True)
    sample = files[0] if files else None
    info = {"pyesef_version": getattr(pyesef, "__version__", "?"),
            "fetched_url": fetched_url, "report_file": sample,
            "n_files_in_zip": len(files),
            "module_dir": [x for x in dir(pyesef) if not x.startswith("_")][:40]}
    if hasattr(pyesef, "extract") and sample:
        try:
            facts = pyesef.extract(sample)
            info["sample_n_facts"] = len(facts) if hasattr(facts, "__len__") else None
        except Exception as e:
            info["extract_error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("pyesef", main)
