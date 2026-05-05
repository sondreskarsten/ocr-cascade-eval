from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from sentence_transformers import SentenceTransformer
    samples = json.loads(open(fx["samples.json"]).read())
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    queries = ["search_query: " + item["q"] for item in samples["queries"]]
    cands = ["search_document: " + c for c in samples["canonical_titles"]]
    qe = model.encode(queries, normalize_embeddings=True)
    ce = model.encode(cands, normalize_embeddings=True)
    s = qe @ ce.T
    top1 = []
    for i, item in enumerate(samples["queries"]):
        idx = int(s[i].argmax())
        top1.append({"q": item["q"], "tier": item["tier"],
                     "top1": samples["canonical_titles"][idx],
                     "score": float(s[i][idx])})
    self_match = sum(1 for r in top1 if r["q"] == r["top1"])
    return {"checkpoint": "nomic-ai/nomic-embed-text-v1.5",
            "self_match_rate": round(self_match/len(queries),3),
            "sample_top1": top1[:30]}


if __name__ == "__main__":
    run_with_metrics("nomic_embed", main)
