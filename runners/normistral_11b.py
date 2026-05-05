from shared import for_each_pdf, run_with_metrics


def main():
    import os
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    candidates = [
        ("norallm/normistral-11b-thinking-gguf", "normistral-11B-thinking-Q4_K_M.gguf"),
        ("norallm/normistral-11b-warm", "normistral-11b-warm.Q4_K_M.gguf"),
        ("norallm/normistral-11b-warm", "normistral-11b-warm.Q3_K_M.gguf"),
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
            f"Her er et utdrag fra et norsk regnskap:\n\n{excerpt}\n\nHva er selskapets navn?\nSvar:",
            f"Her er et utdrag fra et norsk regnskap:\n\n{excerpt}\n\nHva er årsresultatet?\nSvar:",
            f"Her er et utdrag fra et norsk regnskap:\n\n{excerpt}\n\nList opp tre hovedpunkter fra resultatregnskapet.\nSvar:",
        ]
        comps = []
        for p in prompts:
            out = llm(p, max_tokens=200, temperature=0.0, stop=["\n\n\n"])
            comps.append({"prompt_tail": p[-200:], "completion": out["choices"][0]["text"]})
        return {"prompts_run": len(comps), "completions": comps}

    return {"checkpoint_repo": chosen[0], "checkpoint_file": chosen[1],
            "lookup_log": log, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("normistral_11b", main)
