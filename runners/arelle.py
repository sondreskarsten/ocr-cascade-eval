from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import os, urllib.request, zipfile, glob, subprocess

    candidates = [
        "https://filings.xbrl.org/213800FF8F9OG6MTUE31/2024-12-31/ESEF/NO/0/213800FF8F9OG6MTUE31-2024-12-31-ESEF-NO-0.zip",
        "https://filings.xbrl.org/549300SUWCZWERMVB019/2024-12-31/ESEF/NO/0/549300SUWCZWERMVB019-2024-12-31-ESEF-NO-0.zip",
        "https://filings.xbrl.org/2138006O0X73VFNUH294/2023-12-31/ESEF/NO/0/2138006O0X73VFNUH294-2023-12-31-ESEF-NO-0.zip",
    ]
    fetch_log = []
    target_zip, target_dir = "/tmp/esef_sample.zip", "/tmp/esef_sample"
    fetched_url = None
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(target_zip, "wb") as f:
                f.write(resp.read())
            os.makedirs(target_dir, exist_ok=True)
            with zipfile.ZipFile(target_zip, "r") as z:
                z.extractall(target_dir)
            fetched_url = url
            break
        except Exception as e:
            fetch_log.append({url: f"{type(e).__name__}: {e}"})

    if fetched_url is None:
        return {"status": "error", "fetch_log": fetch_log}

    candidate = None
    for ext in (".xhtml", ".html", ".htm"):
        m = glob.glob(f"{target_dir}/**/*{ext}", recursive=True)
        if m:
            candidate = m[0]; break

    out = {"fetched_url": fetched_url, "report_file": candidate}
    proc = subprocess.run(
        ["python", "-m", "arelle.CntlrCmdLine", "-f", candidate or "",
         "--plugins", "validate/ESEF", "--logFile", "/tmp/arelle.log"],
        capture_output=True, text=True, timeout=300,
    )
    out["arelle_rc"] = proc.returncode
    out["arelle_stdout_tail"] = proc.stdout[-2000:]
    out["arelle_stderr_tail"] = proc.stderr[-2000:]
    return out


if __name__ == "__main__":
    run_with_metrics("arelle", main)
