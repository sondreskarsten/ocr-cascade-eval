from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from sentence_transformers import SentenceTransformer

    samples = json.loads(open(fx["samples.json"]).read())
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

    queries = [item["q"] for item in samples["queries"]]
    candidates = samples["canonical_titles"]
    q_emb = model.encode(queries, task="text-matching", normalize_embeddings=True)
    c_emb = model.encode(candidates, task="text-matching", normalize_embeddings=True)
    scores = q_emb @ c_emb.T
    top1 = []
    for i, q in enumerate(queries):
        idx = int(scores[i].argmax())
        top1.append({"q": q, "tier": samples["queries"][i]["tier"],
                     "top1": candidates[idx], "score": float(scores[i][idx])})
    self_match = sum(1 for r in top1 if r["q"] == r["top1"])
    return {"checkpoint": "jinaai/jina-embeddings-v3",
            "license": "CC-BY-NC (non-commercial)",
            "n_queries": len(queries), "n_candidates": len(candidates),
            "self_match_rate": round(self_match/len(queries), 3),
            "sample_top1": top1[:30]}


if __name__ == "__main__":
    run_with_metrics("jina_v3", main)
