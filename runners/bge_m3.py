from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from sentence_transformers import SentenceTransformer
    import numpy as np

    samples = json.loads(open(fx["samples.json"]).read())
    model = SentenceTransformer("BAAI/bge-m3")
    query = samples["norwegian_label"]
    candidates = samples["candidates"]

    q_emb = model.encode([query], normalize_embeddings=True)
    c_emb = model.encode(candidates, normalize_embeddings=True)
    scores = (q_emb @ c_emb.T)[0]
    ranked = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: -x[1])
    return {"checkpoint": "BAAI/bge-m3",
            "dim": int(q_emb.shape[1]),
            "query": query,
            "ranked": ranked}


if __name__ == "__main__":
    run_with_metrics("bge_m3", main)
