from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())
    queries = ["Hva er årsresultat?", "Hva er totale lønnskostnader?",
               "Resultatført skatt på ordinært resultat"]

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", max_length=512)

    def per_pdf(pdf_id, b):
        lines = [ln.strip() for ln in b["full_text"].splitlines()
                 if 3 <= len(ln.strip()) <= 120]
        seen = []
        for ln in lines:
            if ln not in seen: seen.append(ln)
            if len(seen) >= 50: break
        if not seen:
            return {"n_lines": 0}
        per_query = []
        for q in queries:
            pairs = [(q, ln) for ln in seen]
            scores = reranker.predict(pairs).tolist()
            ranked = sorted(zip(seen, scores), key=lambda x: -x[1])
            per_query.append({"q": q,
                              "top5": [(ln, float(s)) for ln, s in ranked[:5]]})
        return {"n_lines": len(seen), "per_query": per_query}

    return {"checkpoint": "mixedbread-ai/mxbai-rerank-large-v1", "queries": queries,
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("mxbai_rerank", main)
