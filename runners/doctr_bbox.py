"""doctr_bbox — preserve doctr's word-level bbox + confidence for cascade voting.

Why this exists: doctr.render() concatenates words with " " which loses
thousands-separator structure on tabular layouts. The underlying model exposes
word-level confidence (`Word.confidence`) and bbox via `Word.geometry`. With
row-aware token clustering — same trick as tesseract TSV — we can recover
multi-token integers like "16 164 967" that the rendered text mode merges
incorrectly.

Empirically, doctr loses Norwegian diacritics (Latin-only model) but its
numeric extraction is competitive with paddleocr/ocrmypdf. This runner adds
confidence-weighted votes.
"""
from shared import for_each_pdf, run_with_metrics
import re


def _cluster_row_numbers(words, x_gap_threshold=250):
    """Same row-clustering as tesseract_tsv: group adjacent number-tokens within a row."""
    by_row = {}
    for w in words:
        # row_id by quantized y-center (5px buckets)
        cy = (w["bbox_abs"][1] + w["bbox_abs"][3]) / 2
        row_id = int(round(cy / 15))
        by_row.setdefault(row_id, []).append(w)

    found = []
    for row_id, ws in by_row.items():
        ws.sort(key=lambda w: w["bbox_abs"][0])
        i = 0
        while i < len(ws):
            t = ws[i]["text"]
            if not re.match(r"^-?\d+$", t):
                i += 1
                continue
            sign = "-" if t.startswith("-") else ""
            digits = re.sub(r"[^\d]", "", t)
            x_anchor = ws[i]["bbox_abs"][2]
            confs = [ws[i]["conf"]]
            j = i + 1
            while j < len(ws):
                tn = ws[j]["text"]
                if (re.match(r"^\d{1,3}$", tn) and
                    ws[j]["bbox_abs"][0] - x_anchor < x_gap_threshold):
                    digits += tn
                    x_anchor = ws[j]["bbox_abs"][2]
                    confs.append(ws[j]["conf"])
                    j += 1
                else:
                    break
            try:
                v = int(sign + digits)
                if abs(v) >= 10:
                    found.append({
                        "value": v,
                        "n_tokens": j - i,
                        "avg_conf": round(sum(confs) / len(confs), 4),
                    })
            except: pass
            i = max(j, i + 1)
    return found


def main():
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    model = ocr_predictor(pretrained=True, det_arch="db_resnet50", reco_arch="crnn_vgg16_bn")

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            doc = DocumentFile.from_images(img_path)
            result = model(doc)
            page_obj = result.pages[0]
            h, w = page_obj.dimensions
            words = []
            for block in page_obj.blocks:
                for line in block.lines:
                    for word in line.words:
                        # geometry is normalized [0,1] coords
                        (x0, y0), (x1, y1) = word.geometry
                        words.append({
                            "text": word.value,
                            "conf": round(float(word.confidence), 4),
                            "bbox_abs": [int(x0 * w), int(y0 * h),
                                          int(x1 * w), int(y1 * h)],
                        })

            numbers = _cluster_row_numbers(words)
            pages.append({
                "page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                "n_words": len(words),
                "n_numbers": len(numbers),
                "n_low_conf_numbers": sum(1 for n in numbers if n["avg_conf"] < 0.85),
                "avg_word_conf": round(sum(w["conf"] for w in words) / max(len(words), 1), 3),
                "words": words,
                "numbers": numbers,
            })

        all_values = sorted({n["value"] for p in pages for n in p["numbers"]})
        return {
            "n_pages": len(pages),
            "n_unique_numbers": len(all_values),
            "all_values": all_values,
            "pages": pages,
        }

    return {"engine": "docTR (db_resnet50 + crnn_vgg16_bn)",
            "signal": "word-level bbox + conf with row-cluster numeric extraction",
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("doctr_bbox", main)
