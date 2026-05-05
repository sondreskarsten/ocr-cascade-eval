from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    from sentence_transformers import SentenceTransformer

    samples = json.loads(open(fx["samples.json"]).read())
    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

    task = "Given a Norwegian financial label, retrieve the canonical English/Norwegian label that matches it."
    query = f"Instruct: {task}\nQuery: {samples['norwegian_label']}"
    candidates = samples["candidates"]
    q_emb = model.encode([query], normalize_embeddings=True)
    c_emb = model.encode(candidates, normalize_embeddings=True)
    scores = (q_emb @ c_emb.T)[0]
    ranked = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: -x[1])
    return {"checkpoint": "intfloat/multilingual-e5-large-instruct",
            "dim": int(q_emb.shape[1]),
            "query": samples["norwegian_label"],
            "ranked": ranked}


if __name__ == "__main__":
    run_with_metrics("multilingual_e5", main)
