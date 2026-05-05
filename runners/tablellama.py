from shared import for_each_pdf, run_with_metrics


def main():
    import os
    from huggingface_hub import hf_hub_download

    candidates = [
        ("MaziyarPanahi/TableLlama-GGUF", "TableLlama.Q4_K_M.gguf"),
        ("TheBloke/TableLlama-GGUF", "tablellama.Q4_K_M.gguf"),
        ("RichardErkhov/osunlp_-_TableLlama-gguf", "TableLlama.Q4_K_M.gguf"),
    ]
    gguf, log = None, []
    for r, fn in candidates:
        try:
            gguf = hf_hub_download(repo_id=r, filename=fn); break
        except Exception as e:
            log.append({f"{r}/{fn}": f"{type(e).__name__}: {e}"})
    if gguf is None:
        return {"status": "error", "lookup_log": log,
                "intended": "osunlp/TableLlama (Llama-2-13B base)"}

    from llama_cpp import Llama
    llm = Llama(model_path=gguf, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)

    def per_pdf(pdf_id, b):
        excerpt = b["full_text"][:1500]
        prompt = f"Question: What is the company name and årsresultat in this Norwegian financial statement?\n{excerpt}\nAnswer:"
        out = llm(prompt, max_tokens=100, temperature=0.0)
        return {"answer": out["choices"][0]["text"]}

    return {"checkpoint": "TableLlama Q4_K_M", "lookup_log": log, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("tablellama", main)
