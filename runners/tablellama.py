from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, os
    from huggingface_hub import hf_hub_download

    samples = json.loads(open(fx["samples.json"]).read())

    candidates = [
        ("MaziyarPanahi/TableLlama-GGUF", "TableLlama.Q4_K_M.gguf"),
        ("TheBloke/TableLlama-GGUF", "tablellama.Q4_K_M.gguf"),
    ]
    gguf, log = None, []
    for r, fn in candidates:
        try:
            gguf = hf_hub_download(repo_id=r, filename=fn)
            break
        except Exception as e:
            log.append({f"{r}/{fn}": f"{type(e).__name__}: {e}"})
    if gguf is None:
        return {"status": "error", "lookup_log": log,
                "intended": "osunlp/TableLlama (Llama-2-13B base)"}

    from llama_cpp import Llama
    llm = Llama(model_path=gguf, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
    prompt = f"Question: {samples['table_qa_question']}\nAnswer:"
    out = llm(prompt, max_tokens=64, temperature=0.0)
    return {"checkpoint": "osunlp/TableLlama (GGUF Q4_K_M)",
            "gguf_path": gguf, "prompt": prompt,
            "completion": out["choices"][0]["text"]}


if __name__ == "__main__":
    run_with_metrics("tablellama", main)
