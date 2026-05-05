from shared import for_each_pdf, run_with_metrics


def main():
    import urllib.request

    info = {"library": "pyesef"}
    try:
        import pyesef
        info["version"] = getattr(pyesef, "__version__", "?")
        info["module_dir"] = sorted([a for a in dir(pyesef) if not a.startswith("_")])[:40]
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"

    sec_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    target = "/tmp/sec_ixbrl.htm"
    try:
        req = urllib.request.Request(sec_url, headers={"User-Agent": "sondre-eval@example.com"})
        with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
            f.write(r.read())
        info["sample_fetched"] = True
    except Exception as e:
        info["sample_fetch_error"] = f"{type(e).__name__}: {e}"

    def per_pdf(pdf_id, b):
        return {"note": "pyesef + iXBRL: Norwegian regnskap PDFs are scans without XBRL tagging.",
                "pdf_chars": len(b["full_text"])}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("pyesef", main)
