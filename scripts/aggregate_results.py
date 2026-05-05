import json, os
from google.cloud import storage

BUCKET = "sondre_brreg_data"
PREFIX = "raw/ocr_eval_2026_05_05/results"


def main():
    cli = storage.Client()
    rows = []
    for blob in cli.list_blobs(BUCKET, prefix=PREFIX):
        if blob.name.endswith(".json"):
            data = json.loads(blob.download_as_bytes())
            rows.append({
                "model": data.get("model"),
                "status": data.get("status"),
                "wall_s": data.get("wall_s"),
                "rss_mb_delta": data.get("rss_mb_delta"),
                "error_type": data.get("error_type"),
                "error_msg": (data.get("error_msg") or "")[:120],
            })
    rows.sort(key=lambda r: (r["status"] != "ok", r["model"] or ""))
    print(f"{'MODEL':30} {'STATUS':8} {'WALL_S':>8} {'ΔRSS_MB':>10} ERROR")
    print("-" * 100)
    for r in rows:
        print(f"{(r['model'] or '?'):30} {(r['status'] or '?'):8} "
              f"{(r['wall_s'] or 0):>8} {(r['rss_mb_delta'] or 0):>10} "
              f"{r['error_type'] or ''} {r['error_msg']}")
    print(f"\n{sum(1 for r in rows if r['status']=='ok')}/{len(rows)} OK")


if __name__ == "__main__":
    main()
