"""Force-redeploy every Cloud Run job (PATCH refreshes the image digest)
and immediately trigger them. Run after a fresh image build."""

import json, os, urllib.request, urllib.error, concurrent.futures
from google.auth.transport.requests import Request
from google.oauth2 import service_account

KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS",
                     "/mnt/project/sondreskarsten-d7d14-8486be2d085b.json")
SPEC = json.loads(open(os.path.join(os.path.dirname(__file__), "..", "jobs.json")).read())
COMMON = SPEC["common"]
JOBS = SPEC["jobs"]
PROJECT = COMMON["project"]
REGION = COMMON["region"]


def token():
    creds = service_account.Credentials.from_service_account_file(
        KEY, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    return creds.token


def http(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, method=method, data=data,
        headers={"Authorization": f"Bearer {token()}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def kick(j):
    base = f"https://run.googleapis.com/v2/projects/{PROJECT}/locations/{REGION}/jobs/{j['name']}"
    code, body = http("GET", base)
    if code != 200:
        return j["name"], "get-failed", code
    # Bump an env var to force new revision -> new image pull, and set RUN_PREFIX
    body["template"]["template"].setdefault("containers", [{}])
    container = body["template"]["template"]["containers"][0]
    env = container.get("env", []) or []
    import time
    run_prefix = os.environ.get("RUN_PREFIX", "raw/ocr_eval_2026_05_05")
    env = [e for e in env if e.get("name") not in ("REBUILD_TS", "RUN_PREFIX")]
    env.append({"name": "REBUILD_TS", "value": str(int(time.time()))})
    env.append({"name": "RUN_PREFIX", "value": run_prefix})
    container["env"] = env
    code_p, _ = http("PATCH", base, body)
    if code_p not in (200, 202):
        return j["name"], f"patch-failed {code_p}", None
    code_r, _ = http("POST", f"{base}:run", {})
    return j["name"], "ok" if code_r in (200, 202) else f"run-failed {code_r}", None


if __name__ == "__main__":
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(kick, JOBS):
            results.append(r)
            print(r)
    ok = sum(1 for r in results if r[1] == "ok")
    print(f"\n{ok}/{len(results)} jobs redeployed + triggered")
