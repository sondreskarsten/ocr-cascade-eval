import json, os, urllib.request, urllib.error
from google.auth.transport.requests import Request as GAuthRequest
from google.oauth2 import service_account

KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS",
                     "/mnt/project/sondreskarsten-d7d14-8486be2d085b.json")
SPEC = json.loads(open(os.path.join(os.path.dirname(__file__), "..", "jobs.json")).read())
COMMON = SPEC["common"]


def token():
    creds = service_account.Credentials.from_service_account_file(
        KEY, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GAuthRequest())
    return creds.token


def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token()}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


if __name__ == "__main__":
    project = COMMON["project"]
    region = COMMON["region"]
    parent = f"projects/{project}/locations/{region}"
    print(f"{'JOB':40} {'LATEST EXEC':30} {'STATE'}")
    print("-" * 90)
    for j in SPEC["jobs"]:
        url = f"https://run.googleapis.com/v2/{parent}/jobs/{j['name']}/executions"
        code, body = get(url)
        execs = body.get("executions", [])
        if not execs:
            print(f"{j['name']:40} {'(none)':30} -")
            continue
        latest = execs[0]
        name = latest.get("name", "").split("/")[-1]
        cond = latest.get("conditions", [])
        state = "running"
        for c in cond:
            if c.get("type") == "Completed":
                state = c.get("state", "?")
                if c.get("reason"):
                    state += f" ({c['reason']})"
        print(f"{j['name']:40} {name:30} {state}")
