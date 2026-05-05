from shared import for_each_pdf, run_with_metrics


def main():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import torch

    ckpt = "NbAiLab/nb-bert-base"
    try:
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForMaskedLM.from_pretrained(ckpt)
        model.eval()
    except Exception as e:
        return {"status": "error", "msg": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        # nb-bert is masked LM. Run a few Norwegian fill-mask probes.
        prompts = [
            "Selskapets [MASK] for 2024 var positivt.",
            "Sum [MASK] i konsernet er 100 millioner kroner.",
            "Resultat før [MASK] var positivt.",
            "Aksjekapital og [MASK] er regulert i selskapets vedtekter.",
        ]
        results = []
        for p in prompts:
            try:
                ids = tok(p, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    logits = model(**ids).logits
                mask_pos = (ids.input_ids[0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    results.append({"prompt": p, "error": "mask not found"})
                    continue
                top5 = logits[0, mask_pos[0]].topk(5)
                results.append({"prompt": p,
                                "top5": [(tok.decode([t.item()]).strip(), float(s))
                                         for t, s in zip(top5.indices, top5.values)]})
            except Exception as e:
                results.append({"prompt": p, "error": f"{type(e).__name__}: {e}"})
        return {"task": "fill-mask", "results": results}

    return {"checkpoint": ckpt, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("nb_bert", main)
