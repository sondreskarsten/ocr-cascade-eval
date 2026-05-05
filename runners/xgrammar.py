from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())
    try:
        import xgrammar as xgr
    except Exception as e:
        return {"status": "error", "library": "xgrammar",
                "import_error": f"{type(e).__name__}: {e}",
                "note": "Usually shipped inside vLLM / SGLang / TensorRT-LLM."}
    canonicals = samples["canonical_titles"][:50]
    schema = {"type": "object",
              "properties": {"canonical": {"type": "string", "enum": canonicals}},
              "required": ["canonical"]}
    out = {"library": "xgrammar", "version": getattr(xgr, "__version__", "?"),
           "schema_demo": schema,
           "n_choices": len(canonicals)}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok))
        grammar = compiler.compile_json_schema(json.dumps(schema))
        out["compiled_grammar"] = "ok"
    except Exception as e:
        out["compile_error"] = f"{type(e).__name__}: {e}"
    return out


if __name__ == "__main__":
    run_with_metrics("xgrammar", main)
