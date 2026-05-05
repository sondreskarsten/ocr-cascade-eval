import json, os, time, concurrent.futures
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


def trigger(j):
    project = COMMON["project"]
    region = COMMON["region"]
    url = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs/{j['name']}:run"
    req = urllib.request.Request(url, method="POST", data=b"{}",
        headers={"Authorization": f"Bearer {token()}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return j["name"], r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return j["name"], e.code, json.loads(e.read() or b"{}")


if __name__ == "__main__":
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        for r in ex.map(trigger, JOBS):
            results.append(r)
            op = r[2].get("name", "")
            print(r[0], r[1], op[-80:] if op else r[2])
    ok = sum(1 for r in results if r[1] in (200, 202))
    print(f"\n{ok}/{len(results)} triggered")
