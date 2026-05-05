from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from rapidfuzz import process, fuzz

    samples = json.loads(open(fx["samples.json"]).read())
    queries = [item["q"] for item in samples["queries"]]
    candidates = samples["canonical_titles"]

    top1 = []
    for i, q in enumerate(queries):
        m = process.extractOne(q, candidates, scorer=fuzz.WRatio)
        top1.append({"q": q, "tier": samples["queries"][i]["tier"],
                     "top1": m[0], "score": m[1]})
    self_match = sum(1 for r in top1 if r["q"] == r["top1"])
    return {"library": "rapidfuzz", "scorer": "WRatio",
            "n_queries": len(queries), "n_candidates": len(candidates),
            "self_match_rate": round(self_match/len(queries), 3),
            "sample_top1": top1[:30]}


if __name__ == "__main__":
    run_with_metrics("rapidfuzz", main)
