from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import subprocess, os
    res = subprocess.run(["nougat", fx["test.pdf"], "-o", "/tmp/nougat_out", "--no-skipping"],
                         capture_output=True, text=True, timeout=600)
    info = {"engine": "nougat", "rc": res.returncode,
            "stderr_tail": res.stderr[-2000:]}
    out_dir = "/tmp/nougat_out"
    if os.path.isdir(out_dir):
        files = sorted(os.listdir(out_dir))
        info["output_files"] = files
        if files:
            with open(os.path.join(out_dir, files[0])) as f:
                info["sample_md"] = f.read()[:2000]
    return info


if __name__ == "__main__":
    run_with_metrics("nougat", main)
