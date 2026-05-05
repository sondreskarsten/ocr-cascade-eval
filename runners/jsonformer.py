from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())
    info = {"library": "jsonformer"}
    try:
        from jsonformer import Jsonformer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        base = "Qwen/Qwen2.5-0.5B"
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")
        schema = {"type": "object",
                  "properties": {"canonical": {"type": "string"},
                                 "confidence": {"type": "number"}}}
        prompt = f"Map this Norwegian financial label: {samples['norwegian_label']}"
        jf = Jsonformer(model, tok, schema, prompt)
        result = jf()
        info["base_model"] = base
        info["result"] = result
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("jsonformer", main)
