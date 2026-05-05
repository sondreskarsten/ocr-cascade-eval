from shared import for_each_pdf, run_with_metrics


def main():
    from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
    import torch

    ckpt = "ltg/norbert3-base"
    try:
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(ckpt, trust_remote_code=True)
        model.eval()
    except Exception as e:
        return {"status": "error", "msg": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        # Norwegian fill-mask: encode first chunk, mask one token, predict
        text = b["full_text"][:1000]
        # Build a few mask tasks from common labels
        prompts = [
            f"Selskapets <mask> for 2024 var positivt.",
            f"Sum <mask> i konsernet er 100 millioner kroner.",
            f"Resultat <mask> skattekostnad var positivt.",
        ]
        results = []
        for p in prompts:
            try:
                ids = tok(p, return_tensors="pt")
                with torch.no_grad():
                    out = model(**ids).logits
                # find mask position
                mask_token = tok.mask_token_id
                if mask_token is None:
                    results.append({"prompt": p, "error": "no mask token"})
                    continue
                pos = (ids.input_ids[0] == mask_token).nonzero(as_tuple=True)[0]
                if len(pos) == 0:
                    results.append({"prompt": p, "error": "mask not found in encoded"})
                    continue
                top5 = out[0, pos[0]].topk(5)
                results.append({"prompt": p,
                                "top5": [(tok.decode([t.item()]), float(s))
                                         for t, s in zip(top5.indices, top5.values)]})
            except Exception as e:
                results.append({"prompt": p, "error": f"{type(e).__name__}: {e}"})
        return {"task": "fill-mask", "results": results}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("norbert", main)
