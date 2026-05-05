from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, os
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    samples = json.loads(open(fx["samples.json"]).read())

    candidates = [
        ("norallm/normistral-11b-thinking-gguf", "normistral-11B-thinking-Q4_K_M.gguf"),
        ("norallm/normistral-11b-warm", "normistral-11b-warm.Q4_K_M.gguf"),
        ("norallm/normistral-11b-warm", "normistral-11b-warm.Q3_K_M.gguf"),
    ]
    gguf_path, log = None, []
    for r, fn in candidates:
        try:
            gguf_path = hf_hub_download(repo_id=r, filename=fn)
            chosen = (r, fn)
            break
        except Exception as e:
            log.append({f"{r}/{fn}": f"{type(e).__name__}: {e}"})

    if gguf_path is None:
        return {"status": "error", "lookup_log": log}

    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
    prompts = [
        "Forklar kort hva 'utsatt skatt' betyr i norsk regnskap.\nSvar:",
        "Hvordan skiller 'betalbar skatt' seg fra 'skattekostnad' i resultatregnskapet?\nSvar:",
    ]
    completions = []
    for p in prompts:
        out = llm(p, max_tokens=200, temperature=0.0, stop=["\n\n"])
        completions.append({"prompt": p, "completion": out["choices"][0]["text"]})
    return {"checkpoint_repo": chosen[0], "checkpoint_file": chosen[1],
            "lookup_log": log, "completions": completions}


if __name__ == "__main__":
    run_with_metrics("normistral_11b", main)
