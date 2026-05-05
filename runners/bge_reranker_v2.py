from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from sentence_transformers import CrossEncoder

    samples = json.loads(open(fx["samples.json"]).read())
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    query = samples["norwegian_label"]
    candidates = samples["candidates"]
    pairs = [(query, c) for c in candidates]
    scores = reranker.predict(pairs).tolist()
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return {"checkpoint": "BAAI/bge-reranker-v2-m3",
            "query": query,
            "ranked": [(c, float(s)) for c, s in ranked]}


if __name__ == "__main__":
    run_with_metrics("bge_reranker_v2", main)
