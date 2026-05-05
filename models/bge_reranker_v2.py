from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from FlagEmbedding import FlagReranker

    samples = json.loads(open(fx["samples.json"]).read())
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
    query = samples["norwegian_label"]
    candidates = samples["candidates"]
    pairs = [[query, c] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    ranked = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: -x[1])
    return {"checkpoint": "BAAI/bge-reranker-v2-m3",
            "query": query,
            "ranked": ranked}


if __name__ == "__main__":
    run_with_metrics("bge_reranker_v2", main)
