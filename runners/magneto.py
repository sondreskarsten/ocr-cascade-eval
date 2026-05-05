from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())

    info = {"approach": "Magneto schema mapping (arXiv:2412.08194)",
            "github": "https://github.com/VIDA-NYU/data-harmonization-magneto"}

    import subprocess, os, sys
    repo_dir = "/tmp/magneto"
    if not os.path.isdir(repo_dir):
        r = subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/VIDA-NYU/data-harmonization-magneto.git", repo_dir],
                           capture_output=True, text=True, timeout=120)
        info["clone_rc"] = r.returncode
        info["clone_stderr"] = r.stderr[-500:]

    sys.path.insert(0, repo_dir)
    try:
        import magneto
        info["import_ok"] = True
        info["module_dir"] = [x for x in dir(magneto) if not x.startswith("_")][:30]
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        info["files_in_repo"] = os.listdir(repo_dir) if os.path.isdir(repo_dir) else []

    def per_pdf(pdf_id, b):
        return {"input_chars": len(b["full_text"]),
                "candidates": samples["canonical_titles"][:5],
                "note": "Magneto requires source-target table inputs not available here"}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("magneto", main)
