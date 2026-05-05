from shared import for_each_pdf, run_with_metrics


def main():
    info = {"library": "colpali-engine"}
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor
        from PIL import Image
        import torch
        ckpt = "vidore/colpali-v1.2"
        proc = ColPaliProcessor.from_pretrained(ckpt)
        model = ColPali.from_pretrained(ckpt, torch_dtype=torch.float32, device_map="cpu").eval()
        info["loaded"] = True
    except Exception as e:
        return {**info, "status": "error", "load_error": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        # Embed page 1 image as visual document
        try:
            img = Image.open(b["page_imgs"][0]).convert("RGB")
            batch = proc.process_images([img]).to("cpu")
            with torch.no_grad():
                emb = model(**batch)
            shape = list(emb.shape)
            return {"page1_embedding_shape": shape,
                    "n_pages_total": b["n_pages"],
                    "note": "ColPali generates multi-vector page embeddings for late-interaction retrieval"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    return {**info, "checkpoint": "vidore/colpali-v1.2", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("colpali", main)
