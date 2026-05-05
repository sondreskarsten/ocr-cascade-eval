import json, os, time, urllib.request, urllib.error
from datetime import datetime, timezone
from google.auth.transport.requests import Request as GAuthRequest
from google.oauth2 import service_account
from google.cloud import storage

KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS",
                     "/secrets/sa.json")
SPEC = json.loads(open("/app/jobs.json").read())
COMMON = SPEC["common"]
JOBS = SPEC["jobs"]
PROJECT = COMMON["project"]
REGION = COMMON["region"]
PARENT = f"projects/{PROJECT}/locations/{REGION}"

MAX_ATTEMPTS = 3
BUCKET = "sondre_brreg_data"
STATUS_PREFIX = "raw/ocr_eval_2026_05_05/supervisor"


def token():
    creds = service_account.Credentials.from_service_account_file(
        KEY, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GAuthRequest())
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


def list_executions(job_name):
    code, body = http("GET", f"https://run.googleapis.com/v2/{PARENT}/jobs/{job_name}/executions")
    if code != 200:
        return []
    return body.get("executions", [])


def trigger(job_name):
    return http("POST", f"https://run.googleapis.com/v2/{PARENT}/jobs/{job_name}:run", {})


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
            return f"running"
    return "running"


def has_result_in_gcs(model_short):
    cli = storage.Client()
    blob = cli.bucket(BUCKET).blob(f"raw/ocr_eval_2026_05_05/results/{model_short}.json")
    return blob.exists()


def main():
    summary = {"checked_at": datetime.now(timezone.utc).isoformat(),
               "jobs": {}, "totals": {"ok": 0, "running": 0, "failed": 0,
                                       "retried": 0, "exhausted": 0, "no_runs": 0}}

    for j in JOBS:
        job_name = j["name"]
        model_short = job_name.replace("ocr-eval-", "").replace("-", "_")
        execs = list_executions(job_name)
        state = execution_state(execs)
        in_gcs = has_result_in_gcs(model_short)

        info = {"state": state, "n_executions": len(execs), "result_in_gcs": in_gcs}

        if "failed" in state and len(execs) < MAX_ATTEMPTS:
            code, _ = trigger(job_name)
            info["retry_triggered"] = code in (200, 202)
            info["attempt_count_before_retry"] = len(execs)
            summary["totals"]["retried"] += 1
        elif "failed" in state:
            summary["totals"]["exhausted"] += 1
        elif state == "ok":
            summary["totals"]["ok"] += 1
        elif state == "no_runs":
            summary["totals"]["no_runs"] += 1
            code, _ = trigger(job_name)
            info["initial_trigger"] = code in (200, 202)
        else:
            summary["totals"]["running"] += 1

        summary["jobs"][job_name] = info

    cli = storage.Client()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cli.bucket(BUCKET).blob(f"{STATUS_PREFIX}/snapshot_{ts}.json").upload_from_string(
        json.dumps(summary, indent=2), content_type="application/json")
    cli.bucket(BUCKET).blob(f"{STATUS_PREFIX}/latest.json").upload_from_string(
        json.dumps(summary, indent=2), content_type="application/json")

    print(json.dumps(summary["totals"]))
    return 0


if __name__ == "__main__":
    main()
