from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())

    info = {"approach": "ReMatch retrieval-augmented schema matching",
            "github": "https://github.com/MohamedYousef-Hassan/ReMatch"}

    import subprocess, os, sys
    repo_dir = "/tmp/rematch"
    if not os.path.isdir(repo_dir):
        r = subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/MohamedYousef-Hassan/ReMatch.git", repo_dir],
                           capture_output=True, text=True, timeout=120)
        info["clone_rc"] = r.returncode
        info["clone_stderr"] = r.stderr[-500:]

    if os.path.isdir(repo_dir):
        info["repo_files"] = os.listdir(repo_dir)[:20]

    def per_pdf(pdf_id, b):
        return {"input_chars": len(b["full_text"]),
                "candidates": samples["canonical_titles"][:5],
                "note": "ReMatch requires source-target schemas; structural smoke test only"}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("rematch", main)
