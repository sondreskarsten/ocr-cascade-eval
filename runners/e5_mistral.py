from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())
    info = {"checkpoint": "intfloat/e5-mistral-7b-instruct", "size_gb": 14}
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(info["checkpoint"])
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return info
    queries = [item["q"] for item in samples["queries"][:10]]
    cands = samples["canonical_titles"][:50]
    qe = model.encode(queries, normalize_embeddings=True)
    ce = model.encode(cands, normalize_embeddings=True)
    s = qe @ ce.T
    info["sample_top1"] = [(queries[i], cands[int(s[i].argmax())], float(s[i].max()))
                            for i in range(len(queries))]
    return info


if __name__ == "__main__":
    run_with_metrics("e5_mistral", main)
