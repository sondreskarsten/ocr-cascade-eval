from shared import for_each_pdf, run_with_metrics


def main():
    info = {"library": "jsonformer"}

    try:
        import transformers
        if not hasattr(transformers, "LogitsWarper"):
            transformers.LogitsWarper = transformers.LogitsProcessor
        from jsonformer import Jsonformer
        info["import_ok"] = True
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}",
                "note": "jsonformer pinned to old transformers; LogitsWarper removed in 4.46+"}

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    base = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")

    schema = {
        "type": "object",
        "properties": {
            "company_name": {"type": "string"},
            "year": {"type": "number"},
            "currency": {"type": "string"},
        },
        "required": ["company_name", "year"]
    }

    def per_pdf(pdf_id, b):
        excerpt = "\n".join(b["full_text"].splitlines()[:8])
        prompt = f"Extract structured info from this Norwegian financial text:\n{excerpt}\n\nJSON:"
        try:
            jf = Jsonformer(model, tok, schema, prompt, max_string_token_length=20)
            out = jf()
            return {"input_excerpt": excerpt[:200], "extracted": out}
        except Exception as e:
            return {"input_excerpt": excerpt[:200], "error": f"{type(e).__name__}: {e}"}

    return {**info, "base_model": base, "schema": schema,
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("jsonformer", main)
