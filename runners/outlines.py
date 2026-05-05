from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())

    import outlines
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    base = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")

    canonicals = samples["canonical_titles"][:50]

    try:
        from outlines import models, generate
        m = models.Transformers(model, tok)
        choice_gen = generate.choice(m, canonicals)
        prompt = f"Map this Norwegian financial label to the closest canonical category:\n{samples['norwegian_label']}\n\nCanonical:"
        ans = choice_gen(prompt)
        return {"library": "outlines", "base_model": base,
                "method": "constrained-choice (cannot hallucinate non-canonical)",
                "prompt": prompt, "answer": ans, "n_choices": len(canonicals)}
    except Exception as e:
        return {"status": "error", "library": "outlines",
                "error_msg": f"{type(e).__name__}: {e}",
                "note": "API surface differs across outlines versions; smoke test only"}


if __name__ == "__main__":
    run_with_metrics("outlines", main)
