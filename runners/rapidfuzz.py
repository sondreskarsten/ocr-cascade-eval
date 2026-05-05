from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())
    canonical = samples["schema_canonical"]

    from rapidfuzz import process, fuzz

    def per_pdf(pdf_id, b):
        seen = []
        for ln in b["full_text"].splitlines():
            ln = ln.strip()
            if 3 <= len(ln) <= 80 and not ln.replace(" ","").isdigit() and ln not in seen:
                seen.append(ln)
            if len(seen) >= 50: break
        matches = []
        for ln in seen:
            m = process.extractOne(ln, canonical, scorer=fuzz.WRatio)
            matches.append({"line": ln, "best": m[0], "score": float(m[1])})
        matches.sort(key=lambda x: -x["score"])
        return {"n_lines": len(seen),
                "avg_top_score": round(sum(m["score"] for m in matches)/max(len(matches),1), 1),
                "top10": matches[:10], "bottom5": matches[-5:]}

    return {"library": "rapidfuzz", "scorer": "WRatio", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("rapidfuzz", main)
