from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())

    info = {"approach": "SCHEMORA schema mapping (EMNLP 2025 industry track)",
            "paper": "https://aclanthology.org/2025.emnlp-industry.120.pdf"}

    import subprocess, os
    repo_dir = "/tmp/schemora"
    candidates_repos = [
        "https://github.com/leozc/schemora",
        "https://github.com/SchemaMatchingLab/schemora",
    ]
    cloned = False
    for url in candidates_repos:
        if os.path.isdir(repo_dir):
            cloned = True
            break
        r = subprocess.run(["git", "clone", "--depth", "1", url, repo_dir],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            cloned = True
            info["repo_url"] = url
            break
    info["cloned"] = cloned
    if cloned:
        info["repo_files"] = os.listdir(repo_dir)[:20]

    def per_pdf(pdf_id, b):
        return {"input_chars": len(b["full_text"]),
                "candidates": samples["canonical_titles"][:5]}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("schemora", main)
