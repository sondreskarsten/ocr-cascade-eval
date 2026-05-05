from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, os, subprocess
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    samples = json.loads(open(fx["samples.json"]).read())

    repo = "norallm/normistral-11b-warm"
    candidates = [
        ("bartowski/norallm_normistral-11b-warm-GGUF", "norallm_normistral-11b-warm-Q4_K_M.gguf"),
        ("norallm/normistral-11b-warm-GGUF", "normistral-11b-warm-q4_K_M.gguf"),
        ("RichardErkhov/norallm_-_normistral-11b-warm-gguf", "normistral-11b-warm.Q4_K_M.gguf"),
    ]
    gguf_path, log = None, []
    for r, fn in candidates:
        try:
            gguf_path = hf_hub_download(repo_id=r, filename=fn)
            break
        except Exception as e:
            log.append({f"{r}/{fn}": f"{type(e).__name__}: {e}"})

    if gguf_path is None:
        return {"status": "error",
                "note": "no public Q4_K_M GGUF of NorMistral-11B located",
                "lookup_log": log,
                "intended_target": repo}

    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
    prompt = "Forklar kort hva 'utsatt skatt' betyr i norsk regnskap.\nSvar:"
    out = llm(prompt, max_tokens=200, temperature=0.0, stop=["\n\n"])
    return {"checkpoint": repo, "gguf_path": gguf_path,
            "prompt": prompt, "completion": out["choices"][0]["text"]}


if __name__ == "__main__":
    run_with_metrics("normistral_11b", main)
