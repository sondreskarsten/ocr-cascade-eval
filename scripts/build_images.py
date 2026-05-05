import json, os, sys, time, urllib.request, urllib.error, concurrent.futures
from google.auth.transport.requests import Request as GAuthRequest
from google.oauth2 import service_account
from google.cloud import storage

KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS",
                     "/mnt/project/sondreskarsten-d7d14-8486be2d085b.json")
PROJECT = "sondreskarsten-d7d14"
CTX_BUCKET = "sondreskarsten-d7d14_cloudbuild"
CTX_OBJECT = "ocr-cascade-eval/context.tar.gz"
REGISTRY = "europe-north1-docker.pkg.dev/sondreskarsten-d7d14/brreg-pipelines"

IMAGES = [
    ("ocr-eval-hf", "huggingface"),
    ("ocr-eval-paddle", "paddle"),
    ("ocr-eval-table", "table"),
    ("ocr-eval-xbrl", "xbrl"),
    ("ocr-eval-llm", "llm"),
    ("ocr-eval-supervisor", "supervisor"),
]


def token():
    creds = service_account.Credentials.from_service_account_file(
        KEY, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GAuthRequest())
    return creds.token


def submit_build(image, image_dir):
    body = {
        "source": {"storageSource": {"bucket": CTX_BUCKET, "object": CTX_OBJECT}},
        "steps": [
            {"name": "gcr.io/cloud-builders/docker",
             "args": ["build", "-t", f"{REGISTRY}/{image}:latest",
                      "-f", f"images/{image_dir}/Dockerfile", "."]},
            {"name": "gcr.io/cloud-builders/docker",
             "args": ["push", f"{REGISTRY}/{image}:latest"]},
        ],
        "images": [f"{REGISTRY}/{image}:latest"],
        "options": {"machineType": "E2_HIGHCPU_8", "diskSizeGb": 100},
        "timeout": "3600s",
    }
    url = f"https://cloudbuild.googleapis.com/v1/projects/{PROJECT}/builds"
    req = urllib.request.Request(url, method="POST",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return resp.get("metadata", {}).get("build", {}).get("id")


def get_build(bid):
    url = f"https://cloudbuild.googleapis.com/v1/projects/{PROJECT}/builds/{bid}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token()}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get_failure_tail(b, tail_chars=4000):
    try:
        bkt = b.get("logsBucket", "").replace("gs://", "")
        log = storage.Client().bucket(bkt).blob(f"log-{b['id']}.txt").download_as_text()
        return log[-tail_chars:]
    except Exception as e:
        return f"(could not fetch log: {e})"


def main():
    builds = {}
    for image, image_dir in IMAGES:
        bid = submit_build(image, image_dir)
        builds[image] = bid
        print(f"submitted {image:25} {bid}")

    open("/tmp/build_ids.json", "w").write(json.dumps(builds, indent=2))

    pending = dict(builds)
    final = {}
    while pending:
        time.sleep(30)
        done = []
        for image, bid in pending.items():
            try:
                b = get_build(bid)
                status = b.get("status")
                print(f"  {image:25} {status}")
                if status in ("SUCCESS", "FAILURE", "TIMEOUT", "CANCELLED",
                              "INTERNAL_ERROR", "EXPIRED"):
                    done.append((image, status, b))
            except Exception as e:
                print(f"  {image:25} poll-error: {e}")
        for image, status, b in done:
            final[image] = {"status": status, "id": pending[image]}
            if status != "SUCCESS":
                final[image]["failure_info"] = b.get("failureInfo")
                final[image]["log_tail"] = get_failure_tail(b)
            del pending[image]

    open("/tmp/build_results.json", "w").write(json.dumps(final, indent=2, default=str))
    print("\n=== build summary ===")
    for image, info in final.items():
        print(f"  {image:25} {info['status']}")
    succ = sum(1 for v in final.values() if v["status"] == "SUCCESS")
    print(f"\n{succ}/{len(final)} images built")
    return 0 if succ == len(final) else 1


if __name__ == "__main__":
    sys.exit(main())
