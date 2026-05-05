from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from sentence_transformers import CrossEncoder
    samples = json.loads(open(fx["samples.json"]).read())
    rer = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", max_length=512)
    query = samples["norwegian_label"]
    cands = samples["candidates"]
    pairs = [(query, c) for c in cands]
    scores = rer.predict(pairs).tolist()
    ranked = sorted(zip(cands, scores), key=lambda x: -x[1])
    return {"checkpoint": "mixedbread-ai/mxbai-rerank-large-v1",
            "query": query,
            "ranked": [(c, float(s)) for c, s in ranked]}


if __name__ == "__main__":
    run_with_metrics("mxbai_rerank", main)
