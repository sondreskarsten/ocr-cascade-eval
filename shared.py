import os, time, json, traceback, gc, psutil
from pathlib import Path
from google.cloud import storage

BUCKET = "sondre_brreg_data"
RUN_PREFIX = "raw/ocr_eval_2026_05_05"
FIXTURE_PREFIX = f"{RUN_PREFIX}/fixture"
RESULTS_PREFIX = f"{RUN_PREFIX}/results"
LOCAL_FIXTURE = Path("/tmp/fixture")
LOCAL_FIXTURE.mkdir(parents=True, exist_ok=True)


def _client():
    return storage.Client()


def fetch_fixture():
    out = {}
    cli = _client()
    bkt = cli.bucket(BUCKET)
    for blob in cli.list_blobs(BUCKET, prefix=FIXTURE_PREFIX):
        local = LOCAL_FIXTURE / Path(blob.name).name
        if not local.exists():
            blob.download_to_filename(str(local))
        out[local.name] = str(local)
    return out


def write_result(model_name: str, payload: dict):
    payload = {**payload, "model": model_name, "wrote_at": time.time()}
    body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    cli = _client()
    blob = cli.bucket(BUCKET).blob(f"{RESULTS_PREFIX}/{model_name}.json")
    blob.upload_from_string(body, content_type="application/json")
    print(f"[{model_name}] wrote {blob.name} ({len(body)} bytes)")


def run_with_metrics(model_name: str, fn):
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.time()
    payload = {"status": "ok"}
    try:
        result = fn()
        payload.update(result if isinstance(result, dict) else {"output": result})
    except Exception as e:
        payload["status"] = "error"
        payload["error_type"] = type(e).__name__
        payload["error_msg"] = str(e)
        payload["traceback"] = traceback.format_exc()
        print(f"[{model_name}] ERROR: {e}")
    finally:
        gc.collect()
        rss1 = proc.memory_info().rss
        payload["wall_s"] = round(time.time() - t0, 2)
        payload["rss_mb_start"] = round(rss0 / 1e6, 1)
        payload["rss_mb_end"] = round(rss1 / 1e6, 1)
        payload["rss_mb_delta"] = round((rss1 - rss0) / 1e6, 1)
        write_result(model_name, payload)
    return payload
