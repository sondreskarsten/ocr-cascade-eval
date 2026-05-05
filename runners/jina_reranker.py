from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from sentence_transformers import CrossEncoder
    samples = json.loads(open(fx["samples.json"]).read())
    rer = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)
    query = samples["norwegian_label"]
    cands = samples["candidates"]
    pairs = [(query, c) for c in cands]
    scores = rer.predict(pairs).tolist()
    ranked = sorted(zip(cands, scores), key=lambda x: -x[1])
    return {"checkpoint": "jinaai/jina-reranker-v2-base-multilingual",
            "license": "CC-BY-NC",
            "query": query,
            "ranked": [(c, float(s)) for c, s in ranked]}


if __name__ == "__main__":
    run_with_metrics("jina_reranker", main)
