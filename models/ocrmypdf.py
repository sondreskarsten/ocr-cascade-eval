from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import subprocess, pdfplumber
    out_pdf = "/tmp/ocrmypdf_out.pdf"
    res = subprocess.run(
        ["ocrmypdf", "-l", "nor", "--force-ocr", "--output-type", "pdf", fx["test.pdf"], out_pdf],
        capture_output=True, text=True, timeout=600,
    )
    pages = {}
    if res.returncode == 0:
        with pdfplumber.open(out_pdf) as pdf:
            for i in [1, 5]:
                txt = pdf.pages[i].extract_text() or ""
                pages[f"p{i+1:02d}"] = {"n_chars": len(txt), "text": txt}
    return {
        "engine": "ocrmypdf+tesseract-nor",
        "stdout_tail": res.stdout[-1000:],
        "stderr_tail": res.stderr[-1000:],
        "rc": res.returncode,
        "pages": pages,
    }


if __name__ == "__main__":
    run_with_metrics("ocrmypdf", main)
