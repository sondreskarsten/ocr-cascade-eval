from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, os
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    samples = json.loads(open(fx["samples.json"]).read())
    candidates = [
        ("bartowski/NorwAI-Magistral-24B-GGUF", "NorwAI-Magistral-24B-Q4_K_M.gguf"),
        ("NorwAI/NorwAI-Magistral-24B-GGUF", "NorwAI-Magistral-24B-q4_K_M.gguf"),
        ("mradermacher/NorwAI-Magistral-24B-GGUF", "NorwAI-Magistral-24B.Q4_K_M.gguf"),
        ("RichardErkhov/NorwAI_-_Magistral-24B-gguf", "Magistral-24B.Q4_K_M.gguf"),
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
                "note": "No public GGUF of NorwAI Magistral 24B located on HF Hub.",
                "lookup_log": log}

    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
    prompt = "Skriv ein kort definisjon av 'fri eigenkapital' på nynorsk.\nSvar:"
    out = llm(prompt, max_tokens=200, temperature=0.0)
    return {"checkpoint": "NorwAI/NorwAI-Magistral-24B (Q4_K_M)",
            "gguf_path": gguf_path, "prompt": prompt,
            "completion": out["choices"][0]["text"]}


if __name__ == "__main__":
    run_with_metrics("norwai_magistral_24b", main)
