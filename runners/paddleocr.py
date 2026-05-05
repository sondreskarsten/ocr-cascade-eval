from shared import for_each_pdf, run_with_metrics


def main():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
                    enable_mkldnn=False, device="cpu")

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            res = ocr.predict(img_path)
            page = res[0]
            pages.append({
                "page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                "n_lines": len(page["rec_texts"]),
                "avg_score": round(sum(page["rec_scores"]) / max(len(page["rec_scores"]), 1), 3),
                "text": "\n".join(page["rec_texts"]),
            })
        n_chars = sum(len(p["text"]) for p in pages)
        return {"n_pages": len(pages), "total_chars": n_chars, "pages": pages}

    return {"checkpoint": "PP-OCRv5_mobile latin", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("paddleocr", main)
