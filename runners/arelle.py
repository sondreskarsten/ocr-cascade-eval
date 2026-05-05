from shared import for_each_pdf, run_with_metrics


def main():
    import os, urllib.request, subprocess

    sec_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    fetch_log = []
    target = "/tmp/sec_ixbrl.htm"
    try:
        req = urllib.request.Request(sec_url, headers={"User-Agent": "sondre-eval@example.com"})
        with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
            f.write(r.read())
    except Exception as e:
        fetch_log.append({sec_url: f"{type(e).__name__}: {e}"})
        return {"status": "error", "fetch_log": fetch_log}

    proc = subprocess.run(
        ["python", "-m", "arelle.CntlrCmdLine", "-f", target,
         "--logFile", "/tmp/arelle.log"],
        capture_output=True, text=True, timeout=300,
    )
    arelle_out = {"sec_url": sec_url, "rc": proc.returncode,
                  "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:]}

    def per_pdf(pdf_id, b):
        return {"note": "Arelle is for iXBRL/XBRL — Norwegian regnskap PDFs are scans without XBRL tagging. Apple 10-K test demonstrates the parsing path.",
                "pdf_chars": len(b["full_text"])}

    return {"engine": "arelle", **arelle_out, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("arelle", main)
