from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import os, urllib.request, zipfile, glob, subprocess

    sample_url = "https://filings.xbrl.org/216-100-0006-50-2022-12-31-ESEF-NO-0/reports/216-100-0006-50-2022-12-31-en.zip"
    target_zip = "/tmp/esef_sample.zip"
    target_dir = "/tmp/esef_sample"
    fetched = False
    err = None
    try:
        req = urllib.request.Request(sample_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(target_zip, "wb") as f:
            f.write(resp.read())
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(target_zip, "r") as z:
            z.extractall(target_dir)
        fetched = True
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    out = {"sample_url": sample_url, "fetched": fetched, "fetch_error": err}
    if not fetched:
        return out

    candidate = None
    for ext in (".html", ".htm", ".xhtml"):
        m = glob.glob(f"{target_dir}/**/*{ext}", recursive=True)
        if m:
            candidate = m[0]
            break
    out["report_file"] = candidate

    proc = subprocess.run(
        ["python", "-m", "arelle.CntlrCmdLine", "-f", candidate or "",
         "--plugins", "validate/ESEF", "--logFile", "/tmp/arelle.log"],
        capture_output=True, text=True, timeout=300,
    )
    out["arelle_rc"] = proc.returncode
    out["arelle_stdout_tail"] = proc.stdout[-1500:]
    out["arelle_stderr_tail"] = proc.stderr[-1500:]
    return out


if __name__ == "__main__":
    run_with_metrics("arelle", main)
