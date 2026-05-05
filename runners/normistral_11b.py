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
    ]
    gguf_path, log = None, []
    chosen = None
    for r, fn in candidates:
        try:
            gguf_path = hf_hub_download(repo_id=r, filename=fn)
            chosen = (r, fn)
            break
        except Exception as e:
            log.append({f"{r}/{fn}": f"{type(e).__name__}: {str(e)[:120]}"})

    if gguf_path is None:
        return {"status": "error", "lookup_log": log}

    try:
        # n_ctx kept modest, no chat_format (avoids Jinja2 template parsing issues)
        llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=os.cpu_count(),
                    verbose=False, chat_format=None)
    except Exception as e:
        return {"status": "error", "lookup_log": log,
                "load_error": f"{type(e).__name__}: {str(e)[:300]}",
                "chosen_repo": chosen[0], "chosen_file": chosen[1]}

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
