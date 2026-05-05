from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json


def main():
    fx = fetch_fixture()
    samples = json.loads(open(fx["samples.json"]).read())
    import os
    from huggingface_hub import hf_hub_download

    candidates = [
        ("mradermacher/TableLlama-GGUF", "TableLlama.Q4_K_M.gguf"),
        ("LoneStriker/TableLlama-GGUF", "TableLlama-Q4_K_M.gguf"),
        ("brittlewis12/TableLlama-GGUF", "tablellama.Q4_K_M.gguf"),
        ("nold/TableLlama-GGUF", "TableLlama.Q4_K_M.gguf"),
    ]
    gguf_path, log = None, []
    for r, fn in candidates:
        try:
            gguf_path = hf_hub_download(repo_id=r, filename=fn)
            chosen = (r, fn)
            break
        except Exception as e:
            log.append({f"{r}/{fn}": f"{type(e).__name__}: {str(e)[:120]}"})
    if gguf_path is None:
        return {"status": "error", "intended": "osunlp/TableLlama (Llama-2-13B base)",
                "lookup_log": log}

    from llama_cpp import Llama
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)

    def per_pdf(pdf_id, b):
        prompt = f"Question: {samples['table_qa_question']}\nAnswer:"
        out = llm(prompt, max_tokens=64, temperature=0.0)
        return {"prompt": prompt, "completion": out["choices"][0]["text"]}

    return {"checkpoint_repo": chosen[0], "checkpoint_file": chosen[1],
            "log": log, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("tablellama", main)
