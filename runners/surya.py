from shared import for_each_pdf, run_with_metrics


def main():
    info = {"library": "surya"}
    try:
        from surya.foundation import FoundationPredictor
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from PIL import Image
        det = DetectionPredictor()
        rec = RecognitionPredictor(FoundationPredictor())
    except Exception as e:
        return {"status": "error", "import_error": f"{type(e).__name__}: {e}"}

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            try:
                img = Image.open(img_path).convert("RGB")
                # Newer surya API expects images as list and langs as list-per-image
                # but languages should be a flat list, not nested
                try:
                    out = rec([img], [["nor", "en"]], det)
                except TypeError:
                    # Older API variant
                    out = rec(images=[img], langs=[["nor","en"]], det_predictor=det)
                except Exception:
                    # Try without languages
                    out = rec([img], None, det)
                page_lines = []
                if out and len(out) > 0:
                    res = out[0]
                    if hasattr(res, "text_lines"):
                        page_lines = res.text_lines
                    elif isinstance(res, list):
                        page_lines = res
                texts = [getattr(ln, "text", str(ln)) for ln in page_lines]
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "n_lines": len(texts), "text": "\n".join(texts)})
            except Exception as e:
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "error": f"{type(e).__name__}: {str(e)[:160]}"})
        return {"n_pages": len(pages),
                "total_chars": sum(len(p.get("text","")) for p in pages),
                "pages": pages}

    return {"engine": "surya", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("surya", main)
