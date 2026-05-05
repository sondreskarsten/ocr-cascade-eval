from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, os
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    samples = json.loads(open(fx["samples.json"]).read())
    candidates = [
        ("NorwAI/NorwAI-Magistral-24B-reasoning", "NorwAI-Magistral-24B-reasoning-Q8_0.gguf"),
        ("charles6huang/NorwAI-Mixtral-8x7B-instruct-Q8_0-GGUF", "norwai-mixtral-8x7b-instruct-q8_0.gguf"),
        ("charles6huang/NorwAI-Mixtral-8x7B-Q8_0-GGUF", "norwai-mixtral-8x7b-q8_0.gguf"),
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
        "Skriv ein kort definisjon av 'fri eigenkapital' på nynorsk.\nSvar:",
        "Forklar kort 'going concern' i norsk regnskapssamanheng.\nSvar:",
    ]
    completions = []
    for p in prompts:
        out = llm(p, max_tokens=200, temperature=0.0)
        completions.append({"prompt": p, "completion": out["choices"][0]["text"]})
    return {"checkpoint_repo": chosen[0], "checkpoint_file": chosen[1],
            "lookup_log": log, "completions": completions}


if __name__ == "__main__":
    run_with_metrics("norwai_magistral_24b", main)
