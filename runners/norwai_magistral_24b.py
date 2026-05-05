from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, os
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    samples = json.loads(open(fx["samples.json"]).read())
    candidates = [
        ("NorwAI/NorwAI-Magistral-24B-reasoning", "NorwAI-Magistral-24B-reasoning-Q8_0.gguf"),
        ("NorwAI/NorwAI-Mistral-7B-instruct", "norwai-mistral-7b-instruct-q4_k_m.gguf"),
        ("NorwAI/NorwAI-Mistral-7B-instruct", "norwai-mistral-7b-instruct-q5_k_m.gguf"),
        ("NorwAI/NorwAI-Mistral-7B", "normistral-7b.Q4_K_M.gguf"),
        ("NorwAI/NorwAI-Llama2-7B", "norllama2-7b.Q4_K_M.gguf"),
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
        llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
    except Exception as e:
        return {"status": "error", "lookup_log": log,
                "load_error": f"{type(e).__name__}: {str(e)[:300]}",
                "chosen_repo": chosen[0], "chosen_file": chosen[1]}

    prompts = [
        "Skriv ein kort definisjon av 'fri eigenkapital' på nynorsk.\nSvar:",
        "Forklar kort 'going concern' i norsk regnskapssamanheng.\nSvar:",
    ]
    completions = []
    for p in prompts:
        try:
            out = llm(p, max_tokens=200, temperature=0.0)
            completions.append({"prompt": p, "completion": out["choices"][0]["text"]})
        except Exception as e:
            completions.append({"prompt": p, "error": f"{type(e).__name__}: {e}"})
    return {"chosen_repo": chosen[0], "chosen_file": chosen[1],
            "lookup_log": log, "completions": completions}


if __name__ == "__main__":
    run_with_metrics("norwai_magistral_24b", main)
