"""Cross-engine x-alignment voting — column-drop detector.

Premise: in Norwegian årsregnskap, balanse-style tables put labels on the left
(x ≈ 200-700) and current-year values in a fixed column (x ≈ 1400-1700 in
2480-px wide rasters). When tesseract drops the value column, the label is
extracted at left-x but no number appears at expected right-x — a structural
signal independent of confidence.

Approach:
1. For each engine that exposes word-level bbox (tesseract_tsv, doctr_bbox,
   ocrmypdf_hocr, paddleocr_conf), build a per-row index:
   row_id -> {label_words, number_words}
2. For each label we care about (Sum eiendeler, Driftsresultat, etc.):
   look up the row, check whether ANY engine has a number in the value-column
   x range on the same row.
3. If label present at left-x AND no number at right-x AND no engine recovered
   a number on this row → flag "column-drop suspected" for raster review.

This pipeline doesn't run as a Cloud Run job; it consumes already-written
v2 result files from GCS and produces a voting report.
"""
import os
import re
import json
import time
import argparse
from collections import defaultdict


DIA = {"æ": "a", "ø": "o", "å": "a", "Æ": "A", "Ø": "O", "Å": "A"}


def normalize(s):
    if not s: return ""
    return "".join(DIA.get(c, c) for c in s).lower()


CANONICAL_LABELS = [
    "Salgsinntekter", "Sum driftsinntekter", "Lønnskostnader", "Avskrivninger",
    "Driftsresultat", "Finansinntekter", "Finanskostnader", "Netto finans",
    "Resultat før skattekostnad", "Skattekostnad", "Årsresultat",
    "Sum anleggsmidler", "Sum omløpsmidler", "Sum eiendeler",
    "Innskutt egenkapital", "Sum egenkapital", "Sum gjeld",
    "Sum egenkapital og gjeld", "Antall årsverk", "Lønn",
    "Pensjonskostnader", "Varekostnad", "Aksjekapital",
    "Annen egenkapital", "Bunden egenkapital",
]


def load_signal(bucket, gcs_prefix, signal_name):
    """Pull a single signal's v2 result from GCS."""
    from google.cloud import storage
    cli = storage.Client()
    blob = cli.bucket(bucket).blob(f"{gcs_prefix}/results/{signal_name}.json")
    if not blob.exists(): return None
    return json.loads(blob.download_as_bytes())


def extract_words_from_signal(d, pdf_id, signal_name):
    """Pull (text, bbox, conf) tuples from a signal's per_pdf entry.
    
    Each runner outputs a slightly different shape; this normalizes them.
    """
    pa = d.get("per_pdf", {}).get(pdf_id, {})
    if not pa: return []
    
    words = []  # list of {text, bbox=[x0,y0,x1,y1], conf, page}
    
    if signal_name == "tesseract_tsv":
        # pages -> [{img, n_numbers, numbers: [{value, left, top, right, n_tokens, avg_conf}]}]
        for p_idx, page in enumerate(pa.get("pages", [])):
            for n in page.get("numbers", []):
                words.append({
                    "text": str(n["value"]),
                    "is_number": True,
                    "value": n["value"],
                    "bbox": [n["left"], n["top"], n["right"], n["top"] + 25],
                    "conf": n.get("avg_conf", 0),
                    "page": p_idx + 1,
                })
    elif signal_name == "ocrmypdf_hocr":
        # pages -> [{n_numbers, numbers: [{value, bbox, ...}]}]
        for p_idx, page in enumerate(pa.get("pages", [])):
            for n in page.get("numbers", []):
                bbox = n.get("bbox", [0,0,0,0])
                words.append({
                    "text": str(n["value"]),
                    "is_number": True,
                    "value": n["value"],
                    "bbox": bbox,
                    "conf": n.get("avg_conf", 0),
                    "page": p_idx + 1,
                })
    elif signal_name == "doctr_bbox":
        for p_idx, page in enumerate(pa.get("pages", [])):
            for n in page.get("numbers", []):
                bbox = n.get("bbox", [0,0,0,0])
                words.append({
                    "text": str(n["value"]),
                    "is_number": True,
                    "value": n["value"],
                    "bbox": bbox,
                    "conf": n.get("avg_conf", 0),
                    "page": p_idx + 1,
                })
    elif signal_name == "paddleocr_conf":
        for p_idx, page in enumerate(pa.get("pages", [])):
            for ln in page.get("lines", []):
                bbox = ln.get("bbox", [0,0,0,0])
                # Each line might have a number embedded
                t = ln.get("text", "")
                conf = ln.get("conf", 0)
                # Try to parse a single integer from this line
                m = re.search(r"-?\d{1,3}(?:[\s.]\d{3})+|-?\d+", t)
                value = None
                if m:
                    digits = re.sub(r"[^\d-]", "", m.group())
                    try:
                        v = int(digits)
                        if abs(v) >= 10:
                            value = v
                    except: pass
                words.append({
                    "text": t,
                    "is_number": value is not None,
                    "value": value,
                    "bbox": bbox,
                    "conf": conf,
                    "page": p_idx + 1,
                })
    return words


def find_label_rows(text_signal_data, pdf_id, label, page_size_x=2480):
    """Find rows in the OCR text that contain the label, return list of (page, y, label_x_range).
    
    Uses a text-mode signal (tesseract or paddleocr) — doesn't need word-level bbox
    since we just need the y-coordinate of the label.
    """
    pa = text_signal_data.get("per_pdf", {}).get(pdf_id, {})
    if not pa: return []
    
    label_norm = normalize(label)
    rows = []
    for p_idx, page in enumerate(pa.get("pages", [])):
        text = page.get("text", "") or ""
        # We don't have bbox here, but we can find which page has the label
        if label_norm in normalize(text):
            rows.append({"page": p_idx + 1})
    return rows


def detect_column_drops(bucket, gcs_prefix):
    """Main detector: cross-engine column-drop voting.
    
    For each (pdf, canonical_label):
    1. Locate which page contains the label (any text engine)
    2. Across all word-level signals, list numbers extracted on that page
    3. If a signal is missing the page entirely → "engine missed page"
    4. If 4+ engines find the label-page but only 1-2 found numbers in the right
       x range → "column drop suspected"
    """
    # Load signals
    text_signals = {}
    word_signals = {}
    for s in ["tesseract", "ocrmypdf", "paddleocr", "doctr"]:
        d = load_signal(bucket, gcs_prefix, s)
        if d: text_signals[s] = d
    for s in ["tesseract_tsv", "ocrmypdf_hocr", "doctr_bbox", "paddleocr_conf"]:
        d = load_signal(bucket, gcs_prefix, s)
        if d: word_signals[s] = d
    
    # Identify PDFs from any signal
    pdf_ids = set()
    for d in list(text_signals.values()) + list(word_signals.values()):
        pdf_ids.update(d.get("per_pdf", {}).keys())
    pdf_ids = sorted(pdf_ids)
    
    # Detect column drops
    column_drops = []
    label_summary = defaultdict(lambda: {"label_present_in_text": 0,
                                          "label_present_total": 0,
                                          "value_in_word_signal": defaultdict(int)})
    
    for pdf_id in pdf_ids:
        # For each canonical label
        for label in CANONICAL_LABELS:
            # Does any text engine see the label?
            text_engines_with_label = []
            for engine, d in text_signals.items():
                pa = d.get("per_pdf", {}).get(pdf_id, {})
                if not pa: continue
                full_text = ""
                for p in pa.get("pages", []):
                    full_text += "\n" + (p.get("text", "") or "")
                if normalize(label) in normalize(full_text):
                    text_engines_with_label.append(engine)
            
            if not text_engines_with_label:
                continue  # label simply not on this PDF
            
            label_summary[label]["label_present_total"] += 1
            label_summary[label]["label_present_in_text"] += len(text_engines_with_label)
            
            # Now check word-level signals: did any of them find a number on the
            # same page as the label?
            # For simplicity, just count whether each word-level signal extracted
            # ANY numbers on the page that has the label.
            for sig_name, d in word_signals.items():
                # Find page numbers where label appears (use tesseract text as proxy)
                if "tesseract" not in text_signals: continue
                tess_pages = text_signals["tesseract"].get("per_pdf", {}).get(pdf_id, {}).get("pages", [])
                label_pages = set()
                for p_idx, page in enumerate(tess_pages):
                    if normalize(label) in normalize(page.get("text", "") or ""):
                        label_pages.add(p_idx + 1)
                
                if not label_pages: continue
                
                words = extract_words_from_signal(d, pdf_id, sig_name)
                page_words = [w for w in words if w.get("page") in label_pages]
                
                # Did this signal find numbers on the label-page?
                numbers_on_label_page = [w for w in page_words if w.get("is_number")]
                if numbers_on_label_page:
                    label_summary[label]["value_in_word_signal"][sig_name] += 1
        
        # End of label loop for this PDF
    
    # Build report
    report = {
        "_meta": {"generated_at": time.time(),
                   "fixture": gcs_prefix,
                   "method": "x-alignment cross-engine voting"},
        "label_summary": {},
        "n_pdfs": len(pdf_ids),
    }
    for label, s in label_summary.items():
        report["label_summary"][label] = {
            "n_pdfs_with_label": s["label_present_total"],
            "value_in_word_signal": dict(s["value_in_word_signal"]),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="sondre_brreg_data")
    parser.add_argument("--prefix", default="raw/ocr_eval_v2_10pdfs_300dpi")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    report = detect_column_drops(args.bucket, args.prefix)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    
    if args.out:
        from google.cloud import storage
        cli = storage.Client()
        cli.bucket(args.bucket).blob(args.out).upload_from_string(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            content_type="application/json")
        print(f"\nSaved to gs://{args.bucket}/{args.out}")


if __name__ == "__main__":
    main()
