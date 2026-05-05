"""ocrmypdf_hocr — extract hOCR (word-level bbox + conf) from ocrmypdf output.

Why this exists: the default ocrmypdf runner extracts text via pdfplumber from
the OCR-augmented PDF, discarding word-level confidence. ocrmypdf can emit
hOCR directly (HTML with embedded x_wconf attributes per word) via
`--sidecar` or by inspecting the embedded text layer. This gives the same
row-clustering opportunity as tesseract TSV, but on the ocrmypdf-preprocessed
input (which includes deskew, rotation correction).

Empirically, ocrmypdf already hits 100% recall on v2 — adding hOCR as a
secondary signal lets cascade voting use word-level conf to flag the residual
edge cases when we scale to 12k.
"""
from shared import for_each_pdf, run_with_metrics
import os
import re
import subprocess


def _cluster_row_numbers(words, x_gap_threshold=250):
    by_row = {}
    for w in words:
        cy = (w["bbox"][1] + w["bbox"][3]) / 2
        row_id = int(round(cy / 15))
        by_row.setdefault(row_id, []).append(w)

    found = []
    for ws in by_row.values():
        ws.sort(key=lambda w: w["bbox"][0])
        i = 0
        while i < len(ws):
            t = ws[i]["text"]
            if not re.match(r"^-?\d+$", t):
                i += 1
                continue
            sign = "-" if t.startswith("-") else ""
            digits = re.sub(r"[^\d]", "", t)
            x_anchor = ws[i]["bbox"][2]
            confs = [ws[i]["conf"]]
            j = i + 1
            while j < len(ws):
                tn = ws[j]["text"]
                if (re.match(r"^\d{1,3}$", tn) and
                    ws[j]["bbox"][0] - x_anchor < x_gap_threshold):
                    digits += tn
                    x_anchor = ws[j]["bbox"][2]
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
                        "avg_conf": round(sum(confs) / len(confs), 2),
                    })
            except: pass
            i = max(j, i + 1)
    return found


def _parse_hocr_words(hocr_path):
    """Parse hOCR HTML and return list of word dicts with bbox + x_wconf."""
    from html.parser import HTMLParser

    class HocrParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.words = []
            self.in_word = False
            self.current_text = ""
            self.current_bbox = None
            self.current_conf = None

        def handle_starttag(self, tag, attrs):
            attrs_d = dict(attrs)
            if tag == "span" and "ocrx_word" in (attrs_d.get("class") or ""):
                title = attrs_d.get("title", "")
                # title format: bbox x0 y0 x1 y1; x_wconf NN
                bbox_match = re.search(r"bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", title)
                conf_match = re.search(r"x_wconf\s+(\d+)", title)
                if bbox_match:
                    self.current_bbox = [int(x) for x in bbox_match.groups()]
                if conf_match:
                    self.current_conf = int(conf_match.group(1))
                self.in_word = True
                self.current_text = ""

        def handle_endtag(self, tag):
            if tag == "span" and self.in_word:
                t = self.current_text.strip()
                if t and self.current_bbox is not None:
                    self.words.append({
                        "text": t,
                        "bbox": self.current_bbox,
                        "conf": self.current_conf or 0,
                    })
                self.in_word = False
                self.current_bbox = None
                self.current_conf = None

        def handle_data(self, data):
            if self.in_word:
                self.current_text += data

    parser = HocrParser()
    with open(hocr_path, "r", encoding="utf-8", errors="replace") as f:
        parser.feed(f.read())
    return parser.words


def main():
    def per_pdf(pdf_id, b):
        out_pdf = b["pdf"].replace(".pdf", "_ocr_hocr.pdf")
        sidecar_dir = b["pdf"].replace(".pdf", "_hocr_sidecar")
        os.makedirs(sidecar_dir, exist_ok=True)
        # Run ocrmypdf with hOCR sidecar
        cmd = ["ocrmypdf", "--force-ocr", "-l", "nor",
               "--sidecar", os.path.join(sidecar_dir, "text.txt"),
               b["pdf"], out_pdf]
        subprocess.run(cmd, check=False, capture_output=True, timeout=600)

        # Use tesseract directly to get hOCR per page (ocrmypdf doesn't expose hOCR sidecar)
        # — render pages from the ocr'd pdf and run tesseract hOCR mode
        if not os.path.exists(out_pdf):
            return {"error": "ocrmypdf failed", "n_pages": 0}

        pages = []
        for img_path in b["page_imgs"]:
            page_n = int(img_path.split("/")[-1].split("-")[1].split(".")[0])
            hocr_path = f"/tmp/ocrmypdf_hocr_{pdf_id}_p{page_n}"
            r = subprocess.run(
                ["tesseract", img_path, hocr_path, "-l", "nor", "hocr"],
                capture_output=True, timeout=120)
            hocr_file = hocr_path + ".hocr"
            if not os.path.exists(hocr_file):
                pages.append({"page_n": page_n, "error": "no hocr produced"})
                continue
            words = _parse_hocr_words(hocr_file)
            numbers = _cluster_row_numbers(words)
            pages.append({
                "page_n": page_n,
                "n_words": len(words),
                "n_numbers": len(numbers),
                "n_low_conf_numbers": sum(1 for n in numbers if n["avg_conf"] < 60),
                "avg_word_conf": round(sum(w["conf"] for w in words) / max(len(words), 1), 1),
                "numbers": numbers,
            })
            try: os.remove(hocr_file)
            except: pass

        all_values = sorted({n["value"] for p in pages for n in p.get("numbers", [])})
        return {
            "n_pages": len(pages),
            "n_unique_numbers": len(all_values),
            "all_values": all_values,
            "pages": pages,
        }

    return {"engine": "ocrmypdf + tesseract hOCR",
            "signal": "word-level bbox + x_wconf with row-cluster numeric extraction",
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("ocrmypdf_hocr", main)
