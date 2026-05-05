import os, time, json, traceback, gc, psutil
from pathlib import Path
from google.cloud import storage

BUCKET = "sondre_brreg_data"
RUN_PREFIX = "raw/ocr_eval_2026_05_05"
FIXTURE_PREFIX = f"{RUN_PREFIX}/fixture"
RESULTS_PREFIX = f"{RUN_PREFIX}/results"
LOCAL_FIXTURE = Path("/tmp/fixture")
LOCAL_FIXTURE.mkdir(parents=True, exist_ok=True)

PDF_IDS = ["pdf_a", "pdf_b"]


def _client():
    return storage.Client()


def fetch_fixture():
    out = {}
    cli = _client()
    for blob in cli.list_blobs(BUCKET, prefix=FIXTURE_PREFIX):
        rel = blob.name[len(FIXTURE_PREFIX) + 1:]
        local = LOCAL_FIXTURE / rel
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local))
        out[rel] = str(local)
    return out


def fetch_pdfs():
    fx = fetch_fixture()
    meta = json.loads(open(fx["pdfs_meta.json"]).read())
    bundles = {}
    for pdf_id in PDF_IDS:
        info = meta[pdf_id]
        pages = sorted(
            [v for k, v in fx.items() if k.startswith(f"{pdf_id}_pages/")],
            key=lambda p: int(Path(p).stem.split("-")[1])
        )
        bundles[pdf_id] = {
            "pdf": fx[f"{pdf_id}.pdf"],
            "pdf_ocr": fx[f"{pdf_id}_ocr.pdf"],
            "n_pages": info["n_pages"],
            "page_imgs": pages,
            "page_text": info["page_text"],
            "page_words": info["page_words"],
            "page_size": info["page_size"],
            "full_text": info["full_text"],
        }
    return bundles


def write_result(model_name, payload):
    payload = {**payload, "model": model_name, "wrote_at": time.time()}
    body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    cli = _client()
    blob = cli.bucket(BUCKET).blob(f"{RESULTS_PREFIX}/{model_name}.json")
    blob.upload_from_string(body, content_type="application/json")
    print(f"[{model_name}] wrote {blob.name} ({len(body)} bytes)")


def run_with_metrics(model_name, fn):
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


def for_each_pdf(fn, isolate_errors=True):
    """Apply fn(pdf_id, pdf_bundle) -> dict to each PDF, returns {pdf_a: ..., pdf_b: ...}"""
    bundles = fetch_pdfs()
    out = {}
    for pdf_id in PDF_IDS:
        if isolate_errors:
            try:
                out[pdf_id] = fn(pdf_id, bundles[pdf_id])
            except Exception as e:
                out[pdf_id] = {"_per_pdf_error": f"{type(e).__name__}: {e}"}
        else:
            out[pdf_id] = fn(pdf_id, bundles[pdf_id])
    return out
