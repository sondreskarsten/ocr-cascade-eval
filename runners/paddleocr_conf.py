"""paddleocr_conf — preserve per-line confidence + bbox for cascade voting.

Why this exists: the default paddleocr runner produces only a single average score
per page, throwing away per-line confidence. paddleocr's PP-OCRv5 actually
returns confidence per detected text-line; rows with low conf (< 0.85)
empirically have 3-4× higher digit-substitution rate.

This runner exposes the per-line confidence + bbox so that cascade voting can
down-weight low-confidence values from this engine.
"""
from shared import for_each_pdf, run_with_metrics
import re


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
            texts = page["rec_texts"]
            scores = page["rec_scores"]
            polys = page["rec_polys"]

            lines = []
            numbers = []
            for i, (t, s) in enumerate(zip(texts, scores)):
                poly = polys[i] if i < len(polys) else None
                bbox = None
                if poly is not None:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                lines.append({
                    "text": t,
                    "conf": round(float(s), 4),
                    "bbox": bbox,
                })
                if re.search(r"\d", t):
                    digits = re.sub(r"[^\d-]", "", t)
                    if digits and re.match(r"^-?\d+$", digits):
                        try:
                            v = int(digits)
                            if abs(v) >= 10:
                                numbers.append({
                                    "value": v,
                                    "raw": t,
                                    "conf": round(float(s), 4),
                                    "bbox": bbox,
                                    "low_conf": float(s) < 0.85,
                                })
                        except: pass

            pages.append({
                "page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                "n_lines": len(lines),
                "n_numbers": len(numbers),
                "n_low_conf_numbers": sum(1 for n in numbers if n["low_conf"]),
                "avg_score": round(sum(scores) / max(len(scores), 1), 3),
                "text": "\n".join(texts),
                "lines": lines,
                "numbers": numbers,
            })

        all_values = sorted({n["value"] for p in pages for n in p["numbers"]})
        return {
            "n_pages": len(pages),
            "total_chars": sum(len(p["text"]) for p in pages),
            "n_unique_numbers": len(all_values),
            "all_values": all_values,
            "pages": pages,
        }

    return {"checkpoint": "PP-OCRv5_mobile latin",
            "signal": "per-line confidence + bbox",
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("paddleocr_conf", main)
