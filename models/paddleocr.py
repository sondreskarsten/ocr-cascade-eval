from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    from paddleocr import PaddleOCR
    from PIL import Image

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
        enable_mkldnn=False,
        device="cpu",
    )

    pages = {}
    for label, key in [("p02", "pages_p02.png"), ("p06", "pages_p06.png")]:
        img = Image.open(fx[key]).convert("RGB")
        img.thumbnail((2000, 3000))
        small_path = f"/tmp/{label}.png"
        img.save(small_path)
        res = ocr.predict(small_path)
        page = res[0]
        pages[label] = {
            "n_lines": len(page["rec_texts"]),
            "avg_score": round(sum(page["rec_scores"]) / max(len(page["rec_scores"]), 1), 3),
            "text": "\n".join(page["rec_texts"]),
        }
    return {"pages": pages, "checkpoint": "PP-OCRv5_mobile latin"}


if __name__ == "__main__":
    run_with_metrics("paddleocr", main)
