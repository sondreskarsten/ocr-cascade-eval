from shared import for_each_pdf, run_with_metrics


def main():
    import os
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    candidates = [
        ("NorwAI/NorwAI-Magistral-24B-reasoning", "NorwAI-Magistral-24B-reasoning-Q8_0.gguf"),
        ("charles6huang/NorwAI-Mixtral-8x7B-instruct-Q8_0-GGUF", "norwai-mixtral-8x7b-instruct-q8_0.gguf"),
        ("charles6huang/NorwAI-Mixtral-8x7B-Q8_0-GGUF", "norwai-mixtral-8x7b-q8_0.gguf"),
    ]
    gguf, log, chosen = None, [], None
    for r, fn in candidates:
        try:
            gguf = hf_hub_download(repo_id=r, filename=fn); chosen = (r, fn); break
        except Exception as e:
            log.append({f"{r}/{fn}": f"{type(e).__name__}: {e}"})
    if gguf is None:
        return {"status": "error", "lookup_log": log}

    llm = Llama(model_path=gguf, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)

    def per_pdf(pdf_id, b):
        excerpt = b["full_text"][:2500]
        prompts = [
            f"Her er eit utdrag frå eit norsk regnskap:\n\n{excerpt}\n\nKva er namnet på selskapet?\nSvar:",
            f"Her er eit utdrag frå eit norsk regnskap:\n\n{excerpt}\n\nKva er årsresultatet?\nSvar:",
        ]
        comps = []
        for p in prompts:
            out = llm(p, max_tokens=200, temperature=0.0)
            comps.append({"prompt_tail": p[-200:], "completion": out["choices"][0]["text"]})
        return {"completions": comps}

    return {"checkpoint_repo": chosen[0], "checkpoint_file": chosen[1],
            "lookup_log": log, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("norwai_magistral_24b", main)
