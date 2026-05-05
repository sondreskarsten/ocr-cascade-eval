from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())
    canonical = samples["canonical_titles"]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", trust_remote_code=True)
    cand_emb = model.encode(canonical, normalize_embeddings=True, batch_size=8)

    def per_pdf(pdf_id, b):
        lines = [ln.strip() for ln in b["full_text"].splitlines()
                 if 3 <= len(ln.strip()) <= 80 and not ln.strip().replace(" ","").isdigit()]
        seen = []
        for ln in lines:
            if ln not in seen: seen.append(ln)
            if len(seen) >= 50: break
        if not seen:
            return {"n_extracted": 0, "matches": []}
        emb = model.encode(seen, normalize_embeddings=True, batch_size=8)
        scores = emb @ cand_emb.T
        matches = []
        for i, ln in enumerate(seen):
            j = int(scores[i].argmax())
            matches.append({"line": ln, "best_canonical": canonical[j], "score": float(scores[i][j])})
        matches.sort(key=lambda x: -x["score"])
        return {"n_extracted": len(seen),
                "avg_top1_score": round(sum(m["score"] for m in matches)/len(matches), 3),
                "top10": matches[:10], "bottom5": matches[-5:]}

    return {"checkpoint": "Snowflake/snowflake-arctic-embed-l-v2.0", "n_canonicals": len(canonical),
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("snowflake_arctic", main)
