from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, torch
    from transformers import AutoTokenizer, AutoModel
    samples = json.loads(open(fx["samples.json"]).read())
    ckpt = "ltg/norbert3-base"
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
    model.eval()
    queries = [item["q"] for item in samples["queries"][:30]]
    cands = samples["canonical_titles"][:100]

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, return_tensors="pt", max_length=64)
        with torch.no_grad():
            out = model(**enc)
        emb = out.last_hidden_state.mean(dim=1)
        return torch.nn.functional.normalize(emb, dim=1)

    qe = embed(queries)
    ce = embed(cands)
    scores = (qe @ ce.T).cpu().numpy()
    top1 = []
    for i, q in enumerate(queries):
        idx = int(scores[i].argmax())
        top1.append({"q": q, "top1": cands[idx], "score": float(scores[i][idx])})
    return {"checkpoint": ckpt, "sample_top1": top1[:20]}


if __name__ == "__main__":
    run_with_metrics("norbert", main)
