"""
Autonomous supervisor for OCR cascade evaluation.

Idempotent operations on each run:
  1. Upsert all 76 Cloud Run Jobs (creates new ones, updates existing)
  2. For each job:
     - state==no_runs    -> trigger initial run
     - state==failed and attempts<MAX_ATTEMPTS -> trigger retry
     - state==exhausted (failed and attempts>=MAX) -> leave alone
     - state==ok or running -> leave alone
  3. Write snapshot to gs://sondre_brreg_data/raw/ocr_eval_2026_05_05/supervisor/

Designed to be triggered on a schedule (Cloud Scheduler -> Cloud Run Jobs API).
"""
import json, os, urllib.request, urllib.error
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.auth.transport.requests import Request as GAuthRequest
from google.oauth2 import service_account
from google.cloud import storage

KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
SPEC = json.loads(open("/app/jobs.json").read())
COMMON = SPEC["common"]
JOBS = SPEC["jobs"]
PROJECT, REGION = COMMON["project"], COMMON["region"]
PARENT = f"projects/{PROJECT}/locations/{REGION}"
MAX_ATTEMPTS = 3
BUCKET = "sondre_brreg_data"
RUN_PREFIX = os.environ.get("RUN_PREFIX", "raw/ocr_eval_2026_05_05")
STATUS_PREFIX = f"{RUN_PREFIX}/supervisor"
RESULTS_PREFIX = f"{RUN_PREFIX}/results"


def token():
    if KEY and os.path.exists(KEY):
        creds = service_account.Credentials.from_service_account_file(
            KEY, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    else:
        # Use ambient ADC (Cloud Run service account)
        import google.auth
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GAuthRequest())
    return creds.token


def http(method, url, body=None, tok=None):
    data = json.dumps(body).encode() if body is not None else None
    if tok is None:
        tok = token()
    req = urllib.request.Request(url, method=method, data=data,
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def job_spec(j):
    image = f"{COMMON['registry_prefix']}/{j['image']}:latest"
    return {
        "launchStage": "GA",
        "template": {
            "parallelism": 1, "taskCount": 1,
            "template": {
                "containers": [{
                    "image": image,
                    "args": [f"runners.{j['runner']}"],
                    "resources": {"limits": {"cpu": j["cpu"], "memory": j["memory"]}},
                    "env": [{"name": "RUN_PREFIX", "value": RUN_PREFIX}],
                }],
                "timeout": j["timeout"],
                "serviceAccount": COMMON["service_account"],
                "maxRetries": 0,
            }
        }
    }


def upsert(j, tok):
    base = f"https://run.googleapis.com/v2/{PARENT}/jobs"
    code_get, _ = http("GET", f"{base}/{j['name']}", tok=tok)
    if code_get == 404:
        c, _ = http("POST", f"{base}?jobId={j['name']}", job_spec(j), tok=tok)
        return ("create", c)
    else:
        c, _ = http("PATCH", f"{base}/{j['name']}", job_spec(j), tok=tok)
        return ("update", c)


def list_executions(job_name, tok):
    code, body = http("GET",
        f"https://run.googleapis.com/v2/{PARENT}/jobs/{job_name}/executions", tok=tok)
    if code != 200:
        return []
    return body.get("executions", [])


def trigger(job_name, tok):
    return http("POST", f"https://run.googleapis.com/v2/{PARENT}/jobs/{job_name}:run", {}, tok=tok)


def execution_state(execs):
    if not execs:
        return "no_runs"
    latest = execs[0]
    for c in latest.get("conditions", []):
        if c.get("type") == "Completed":
            state = c.get("state", "?")
            reason = c.get("reason") or ""
            if state == "CONDITION_SUCCEEDED":
                return "ok"
            if state == "CONDITION_FAILED":
                return f"failed:{reason or 'unknown'}"
    return "running"


def has_result(runner_name):
    blob = storage.Client().bucket(BUCKET).blob(
        f"{RESULTS_PREFIX}/{runner_name}.json")
    return blob.exists()


def process_one(j, tok):
    upsert_action, upsert_code = upsert(j, tok)
    execs = list_executions(j["name"], tok)
    state = execution_state(execs)
    result_in_gcs = has_result(j["runner"])
    info = {"runner": j["runner"], "state": state, "n_executions": len(execs),
            "upsert": f"{upsert_action}:{upsert_code}",
            "result_in_gcs": result_in_gcs}
    # If a result file exists, treat as ok regardless of execution state.
    # Some jobs OOM after writing the result, leaving execution_state="failed"
    # while the actual run output is good.
    if result_in_gcs:
        info["effective_state"] = "ok"
        return j["name"], info
    if "failed" in state and len(execs) < MAX_ATTEMPTS:
        c, _ = trigger(j["name"], tok)
        info["retry_triggered"] = c in (200, 202)
        info["retry_status"] = c
    elif "failed" in state:
        info["exhausted"] = True
    elif state == "no_runs":
        c, _ = trigger(j["name"], tok)
        info["initial_trigger"] = c in (200, 202)
        info["initial_status"] = c
    return j["name"], info


def main():
    tok = token()
    summary = {"checked_at": datetime.now(timezone.utc).isoformat(),
               "n_jobs": len(JOBS),
               "jobs": {},
               "totals": {"ok": 0, "running": 0, "no_runs": 0,
                          "retried": 0, "exhausted": 0}}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(process_one, j, tok) for j in JOBS]
        for f in as_completed(futures):
            try:
                name, info = f.result()
                summary["jobs"][name] = info
                if info.get("effective_state") == "ok" or info["state"] == "ok":
                    summary["totals"]["ok"] += 1
                elif info["state"] == "running":
                    summary["totals"]["running"] += 1
                elif info["state"] == "no_runs":
                    summary["totals"]["no_runs"] += 1
                elif info.get("exhausted"):
                    summary["totals"]["exhausted"] += 1
                elif info.get("retry_triggered"):
                    summary["totals"]["retried"] += 1
            except Exception as e:
                print(f"task error: {e}")

    cli = storage.Client()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cli.bucket(BUCKET).blob(f"{STATUS_PREFIX}/snapshot_{ts}.json").upload_from_string(
        json.dumps(summary, indent=2), content_type="application/json")
    cli.bucket(BUCKET).blob(f"{STATUS_PREFIX}/latest.json").upload_from_string(
        json.dumps(summary, indent=2), content_type="application/json")
    print(json.dumps(summary["totals"]))


if __name__ == "__main__":
    main()
