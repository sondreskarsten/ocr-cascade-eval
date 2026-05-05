import json, os, sys, time, concurrent.futures
import urllib.request, urllib.error
from google.auth.transport.requests import Request as GAuthRequest
from google.oauth2 import service_account

KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS",
                     "/mnt/project/sondreskarsten-d7d14-8486be2d085b.json")
SPEC = json.loads(open(os.path.join(os.path.dirname(__file__), "..", "jobs.json")).read())
COMMON = SPEC["common"]
JOBS = SPEC["jobs"]


def token():
    creds = service_account.Credentials.from_service_account_file(
        KEY, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GAuthRequest())
    return creds.token


def request(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, method=method, data=data,
        headers={"Authorization": f"Bearer {token()}",
                 "Content-Type": "application/json"})
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
            "parallelism": 1,
            "taskCount": 1,
            "template": {
                "containers": [{
                    "image": image,
                    "args": [f"runners.{j['runner']}"],
                    "resources": {"limits": {"cpu": j["cpu"], "memory": j["memory"]}},
                }],
                "timeout": j["timeout"],
                "serviceAccount": COMMON["service_account"],
                "maxRetries": 0,
            }
        }
    }


def upsert(j):
    region = COMMON["region"]
    project = COMMON["project"]
    parent = f"projects/{project}/locations/{region}"
    base = f"https://run.googleapis.com/v2/{parent}/jobs"
    body = job_spec(j)
    code_get, _ = request("GET", f"{base}/{j['name']}")
    if code_get == 404:
        code, resp = request("POST", f"{base}?jobId={j['name']}", body)
        action = "create"
    else:
        code, resp = request("PATCH",
            f"https://run.googleapis.com/v2/{parent}/jobs/{j['name']}", body)
        action = "update"
    return j["name"], action, code, resp


if __name__ == "__main__":
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        for r in ex.map(upsert, JOBS):
            results.append(r)
            print(r[:3])
    ok = sum(1 for r in results if r[2] in (200, 201, 202))
    print(f"\n{ok}/{len(results)} OK")
