from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, torch
    from transformers import AutoTokenizer, AutoModel
    samples = json.loads(open(fx["samples.json"]).read())
    ckpt = "NbAiLab/nb-bert-base"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt)
    model.eval()
    queries = [item["q"] for item in samples["queries"]]
    cands = samples["canonical_titles"]

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, return_tensors="pt", max_length=64)
        with torch.no_grad():
            out = model(**enc)
        emb = out.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(emb, dim=1)

    qe = embed(queries)
    ce = embed(cands)
    scores = (qe @ ce.T).cpu().numpy()
    top1 = []
    for i, q in enumerate(queries):
        idx = int(scores[i].argmax())
        top1.append({"q": q, "tier": samples["queries"][i]["tier"],
                     "top1": cands[idx], "score": float(scores[i][idx])})
    self_match = sum(1 for r in top1 if r["q"] == r["top1"])
    return {"checkpoint": ckpt, "self_match_rate": round(self_match/len(queries),3),
            "sample_top1": top1[:30]}


if __name__ == "__main__":
    run_with_metrics("nb_bert", main)
