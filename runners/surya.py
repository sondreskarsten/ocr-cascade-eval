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
                lines = rec([img], [["nor", "en"]], det)
                page_lines = lines[0].text_lines if lines else []
                texts = [ln.text for ln in page_lines]
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "n_lines": len(texts), "text": "\n".join(texts)})
            except Exception as e:
                pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                              "error": f"{type(e).__name__}: {e}"})
        return {"n_pages": len(pages),
                "total_chars": sum(len(p.get("text","")) for p in pages),
                "pages": pages}

    return {"engine": "surya", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("surya", main)
