from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, os
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    candidates = [
        ("norallm/normistral-11b-thinking-gguf", "normistral-11B-thinking-Q4_K_M.gguf"),
        ("norallm/normistral-11b-warm", "normistral-11b-warm.Q4_K_M.gguf"),
        ("norallm/normistral-11b-warm", "normistral-11b-warm.Q3_K_M.gguf"),
        ("norallm/normistral-7b-warm-instruct-gguf", "normistral-7B-warm-instruct-Q4_K_M.gguf"),
        ("norallm/normistral-7b-warm", "normistral-7b-warm.Q4_K_M.gguf"),
    ]
    gguf_path, log = None, []
    chosen = None
    llm = None

    def try_load(p):
        # Strategy 1: plain init, no chat_format (model still parses embedded template)
        try:
            return Llama(model_path=p, n_ctx=2048, n_threads=os.cpu_count(),
                         verbose=False, chat_format=None)
        except Exception as e1:
            err1 = f"plain: {type(e1).__name__}: {str(e1)[:120]}"
        # Strategy 2: override broken template by passing explicit chat_format
        try:
            return Llama(model_path=p, n_ctx=2048, n_threads=os.cpu_count(),
                         verbose=False, chat_format="chatml")
        except Exception as e2:
            err2 = f"chatml: {type(e2).__name__}: {str(e2)[:120]}"
        # Strategy 3: skip metadata-driven template entirely
        try:
            return Llama(model_path=p, n_ctx=2048, n_threads=os.cpu_count(),
                         verbose=False, chat_format=None,
                         tokenizer=None)
        except Exception as e3:
            err3 = f"no_tok: {type(e3).__name__}: {str(e3)[:120]}"
        raise RuntimeError(f"{err1} | {err2} | {err3}")

    for r, fn in candidates:
        try:
            gguf_path = hf_hub_download(repo_id=r, filename=fn)
            chosen = (r, fn)
        except Exception as e:
            log.append({f"download:{r}/{fn}": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        try:
            llm = try_load(gguf_path)
            break  # success
        except Exception as e:
            log.append({f"load:{r}/{fn}": str(e)[:300]})
            llm = None

    if llm is None:
        return {"status": "error", "lookup_log": log,
                "load_error": "all candidates failed (template parsing)",
                "chosen_repo": chosen[0] if chosen else None,
                "chosen_file": chosen[1] if chosen else None}

    # Plain text prompts, no special tokens
    prompts = [
        "Spørsmål: Hva betyr 'utsatt skatt' i norsk regnskap? Svar:",
        "Spørsmål: Hva er forskjellen mellom 'betalbar skatt' og 'skattekostnad'? Svar:",
    ]
    completions = []
    for p in prompts:
        try:
            out = llm(p, max_tokens=180, temperature=0.0,
                      stop=["\n\n", "Spørsmål:"])
            completions.append({"prompt": p, "completion": out["choices"][0]["text"]})
        except Exception as e:
            completions.append({"prompt": p, "error": f"{type(e).__name__}: {e}"})
    return {"chosen_repo": chosen[0], "chosen_file": chosen[1],
            "lookup_log": log, "completions": completions}


if __name__ == "__main__":
    run_with_metrics("normistral_11b", main)
