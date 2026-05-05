from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import os, urllib.request, subprocess

    sample_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    target = "/tmp/aapl-10k.htm"
    try:
        req = urllib.request.Request(sample_url,
            headers={"User-Agent": "Sondre Skarsten sondre@example.no"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as f:
            f.write(resp.read())
        size = os.path.getsize(target)
    except Exception as e:
        return {"status": "error", "fetch_error": f"{type(e).__name__}: {e}",
                "sample_url": sample_url}

    proc = subprocess.run(
        ["python", "-m", "arelle.CntlrCmdLine", "-f", target,
         "--logFile", "/tmp/arelle.log",
         "--factsFile", "/tmp/arelle_facts.csv"],
        capture_output=True, text=True, timeout=600,
    )
    facts_count = 0
    facts_sample = []
    if os.path.exists("/tmp/arelle_facts.csv"):
        with open("/tmp/arelle_facts.csv") as f:
            lines = f.readlines()
            facts_count = max(len(lines) - 1, 0)
            facts_sample = lines[:5]
    return {"sample_url": sample_url, "fetched_bytes": size,
            "arelle_rc": proc.returncode,
            "arelle_stdout_tail": proc.stdout[-1500:],
            "arelle_stderr_tail": proc.stderr[-1500:],
            "facts_count": facts_count, "facts_sample": facts_sample}


if __name__ == "__main__":
    run_with_metrics("arelle", main)
