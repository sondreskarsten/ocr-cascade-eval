from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from sentence_transformers import SentenceTransformer
    samples = json.loads(open(fx["samples.json"]).read())
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    queries = [item["q"] for item in samples["queries"]]
    cands = samples["canonical_titles"]
    qe = model.encode(queries, normalize_embeddings=True)
    ce = model.encode(cands, normalize_embeddings=True)
    s = qe @ ce.T
    top1 = []
    for i, q in enumerate(queries):
        idx = int(s[i].argmax())
        top1.append({"q": q, "tier": samples["queries"][i]["tier"],
                     "top1": cands[idx], "score": float(s[i][idx])})
    self_match = sum(1 for r in top1 if r["q"] == r["top1"])
    return {"checkpoint": "mixedbread-ai/mxbai-embed-large-v1",
            "self_match_rate": round(self_match/len(queries),3),
            "sample_top1": top1[:30]}


if __name__ == "__main__":
    run_with_metrics("mxbai_embed", main)
