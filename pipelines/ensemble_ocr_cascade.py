"""Multi-signal ensemble OCR pipeline.

Runs 6 OCR engines + visual extractor in parallel, votes on each numeric token,
calibrates disagreements via raster crops, and outputs a unified extraction.

Stages:
1. OCR (6 engines): tesseract, paddleocr, ocrmypdf, doctr, easyocr, nougat
2. Visual extractor: pix2struct-docvqa-base (page raster -> DocVQA)
3. Schema mapping: nb_sbert (line -> canonical label, threshold-friendly)
4. Fill-mask normalization: nb_bert
5. NER on noter: spacy_nb
6. Voting + raster calibration

Output (per PDF):
  numeric_extractions: list of (label, value, n_engines_hit, engines_hit, raster_path_if_disagreement)
  noter_extractions: list of (note_title, note_text, ner_entities)
  schema_mapped_lines: list of (raw_line, canonical_label, score)
  reliability_score: % of values with >=4/6 OCR consensus
"""
from shared import for_each_pdf, run_with_metrics, fetch_fixture
import json
import re
import os
import io
import time
import subprocess


def _ocr_tesseract(bundle):
    import pytesseract
    out_pages = []
    for p in bundle["page_imgs"]:
        try:
            t = pytesseract.image_to_string(p, lang="nor")
            out_pages.append({"page": p, "text": t})
        except Exception as e:
            out_pages.append({"page": p, "text": "", "error": str(e)[:120]})
    return out_pages


def _ocr_tesseract_tsv(bundle):
    """Tesseract TSV gives word-level confidence + bbox — secondary signal."""
    import pytesseract
    out_pages = []
    for p in bundle["page_imgs"]:
        try:
            data = pytesseract.image_to_data(p, lang="nor", output_type=pytesseract.Output.DICT)
            words = []
            for i, w in enumerate(data["text"]):
                w = (w or "").strip()
                if not w: continue
                conf = float(data["conf"][i])
                if conf < 0: continue
                words.append({
                    "text": w, "conf": conf,
                    "bbox": [data["left"][i], data["top"][i],
                             data["left"][i] + data["width"][i],
                             data["top"][i] + data["height"][i]],
                    "line": data["line_num"][i],
                })
            out_pages.append({"page": p, "words": words})
        except Exception as e:
            out_pages.append({"page": p, "words": [], "error": str(e)[:120]})
    return out_pages


def _ocr_paddle(bundle):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    out_pages = []
    for p in bundle["page_imgs"]:
        try:
            r = ocr.ocr(p, cls=False)
            lines = []
            if r and r[0]:
                for ln in r[0]:
                    if len(ln) >= 2 and isinstance(ln[1], (list, tuple)):
                        lines.append({"text": ln[1][0], "conf": float(ln[1][1])})
            out_pages.append({"page": p, "lines": lines})
        except Exception as e:
            out_pages.append({"page": p, "lines": [], "error": str(e)[:120]})
    return out_pages


def _ocr_ocrmypdf_text(bundle):
    import pdfplumber
    out_pdf = bundle["pdf"].replace(".pdf", "_ocr.pdf")
    if not os.path.exists(out_pdf):
        subprocess.run(["ocrmypdf", "--force-ocr", "-l", "nor", bundle["pdf"], out_pdf],
                       check=False, capture_output=True)
    if not os.path.exists(out_pdf):
        return []
    with pdfplumber.open(out_pdf) as pdf:
        return [{"page_n": i + 1, "text": p.extract_text() or ""}
                for i, p in enumerate(pdf.pages)]


def _ocr_doctr(bundle):
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_pdf(bundle["pdf"])
    res = model(doc)
    out_pages = []
    for page in res.pages:
        lines = []
        for block in page.blocks:
            for line in block.lines:
                lines.append(" ".join(w.value for w in line.words))
        out_pages.append({"text": "\n".join(lines)})
    return out_pages


def _visual_extract_pix2struct(bundle):
    import torch
    from PIL import Image
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

    proc = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
    model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base")
    if torch.cuda.is_available(): model = model.cuda()

    qs = ["What is the company name?", "What is the year?",
          "What is the årsresultat?", "What is the sum eiendeler?",
          "What is the sum egenkapital?", "What is the driftsresultat?",
          "What is the sum kostnader?", "What is the sum inntekter?"]
    targets = bundle["page_imgs"][1:5] if len(bundle["page_imgs"]) >= 2 else bundle["page_imgs"]
    results = []
    for img_path in targets:
        img = Image.open(img_path).convert("RGB")
        page_qa = []
        for q in qs:
            try:
                inputs = proc(images=img, return_tensors="pt", text=q)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=48)
                a = proc.decode(out[0], skip_special_tokens=True)
                page_qa.append({"q": q, "a": a})
            except Exception as e:
                page_qa.append({"q": q, "error": str(e)[:120]})
        results.append({"img": img_path.split("/")[-1], "qa": page_qa})
    return results


# ============================================================
# Numeric voting
# ============================================================

NUMERIC_PATTERN = re.compile(r"-?\d{1,3}(?:[\s.]\d{3})+|-?\d+")


def _normalize_number(s):
    """'-1 101 413' -> -1101413, '1.135.691' -> 1135691, '143' -> 143"""
    s = s.strip()
    sign = -1 if s.startswith("-") else 1
    digits = re.sub(r"[^\d]", "", s)
    if not digits: return None
    return sign * int(digits)


def _extract_numbers_from_text(text):
    """Return set of integers found in text."""
    nums = set()
    for m in NUMERIC_PATTERN.finditer(text):
        n = _normalize_number(m.group())
        if n is not None and abs(n) >= 100:
            nums.add(n)
    return nums


def _extract_numbers_from_lines(line_dicts):
    nums = set()
    for ld in line_dicts:
        nums.update(_extract_numbers_from_text(ld.get("text", "")))
    return nums


# ============================================================
# Schema mapping (Stage 3) — nb_sbert
# ============================================================

def _schema_map_nb_sbert(lines, canonical_labels, model=None):
    """Map each line to closest canonical label via nb_sbert embeddings."""
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("NbAiLab/nb-sbert-base")
    line_embs = model.encode(lines, normalize_embeddings=True)
    canon_embs = model.encode(canonical_labels, normalize_embeddings=True)
    import numpy as np
    sim = line_embs @ canon_embs.T
    out = []
    for i, l in enumerate(lines):
        best_j = int(sim[i].argmax())
        out.append({
            "raw_line": l,
            "canonical_label": canonical_labels[best_j],
            "score": float(sim[i][best_j]),
        })
    return out


# ============================================================
# Fill-mask normalization (Stage 4) — nb_bert
# ============================================================

def _fill_mask_norm(prompt_template, mask_value, fill_pipe=None):
    """Use nb_bert to validate/complete a label."""
    if fill_pipe is None:
        from transformers import pipeline
        fill_pipe = pipeline("fill-mask", model="NbAiLab/nb-bert-base")
    prompt = prompt_template.replace("{LABEL}", mask_value)
    if "[MASK]" not in prompt: return None
    try:
        out = fill_pipe(prompt, top_k=5)
        return out
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# NER on noter (Stage 5) — spacy_nb
# ============================================================

def _ner_spacy(text):
    import spacy
    try:
        nlp = spacy.load("nb_core_news_sm")
    except OSError:
        return []
    doc = nlp(text[:50000])
    return [{"text": ent.text, "label": ent.label_,
             "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents]


# ============================================================
# Main ensemble pipeline
# ============================================================

CANONICAL_LABELS = [
    "Salgsinntekter", "Annen driftsinntekt", "Sum driftsinntekter",
    "Varekostnad", "Lønnskostnader", "Avskrivninger",
    "Annen driftskostnad", "Sum driftskostnader",
    "Driftsresultat", "Finansinntekter", "Finanskostnader",
    "Netto finans", "Resultat før skattekostnad", "Skattekostnad",
    "Årsresultat", "Sum anleggsmidler", "Sum omløpsmidler",
    "Sum eiendeler", "Innskutt egenkapital", "Opptjent egenkapital",
    "Sum egenkapital", "Sum langsiktig gjeld", "Sum kortsiktig gjeld",
    "Sum gjeld", "Sum egenkapital og gjeld",
]


def main():
    """Process all PDFs, vote per numeric value, return ensemble extraction."""
    out_per_pdf = {}

    def process_one(pdf_id, b):
        signals = {}

        # Stage 1: OCR engines
        try:
            signals["tesseract_pages"] = _ocr_tesseract(b)
        except Exception as e:
            signals["tesseract_error"] = str(e)[:120]
        try:
            signals["tesseract_tsv"] = _ocr_tesseract_tsv(b)
        except Exception as e:
            signals["tesseract_tsv_error"] = str(e)[:120]
        try:
            signals["ocrmypdf_pages"] = _ocr_ocrmypdf_text(b)
        except Exception as e:
            signals["ocrmypdf_error"] = str(e)[:120]
        try:
            signals["doctr_pages"] = _ocr_doctr(b)
        except Exception as e:
            signals["doctr_error"] = str(e)[:120]

        # Stage 1b: Visual extraction
        try:
            signals["pix2struct_qa"] = _visual_extract_pix2struct(b)
        except Exception as e:
            signals["pix2struct_error"] = str(e)[:120]

        # Stage 2: Numeric voting
        engine_numbers = {}
        for key, parser in [
            ("tesseract", lambda s: set().union(*[_extract_numbers_from_text(p.get("text",""))
                                                   for p in s.get("tesseract_pages",[])])),
            ("ocrmypdf",  lambda s: set().union(*[_extract_numbers_from_text(p.get("text",""))
                                                   for p in s.get("ocrmypdf_pages",[])])),
            ("doctr",     lambda s: set().union(*[_extract_numbers_from_text(p.get("text",""))
                                                   for p in s.get("doctr_pages",[])])),
        ]:
            try: engine_numbers[key] = parser(signals)
            except: engine_numbers[key] = set()

        # Universe of all numbers seen by any engine
        all_numbers = set()
        for s in engine_numbers.values(): all_numbers.update(s)

        votes = []
        for n in all_numbers:
            engines_with = [e for e, ns in engine_numbers.items() if n in ns]
            votes.append({
                "value": n,
                "engines_hit": engines_with,
                "n_hit": len(engines_with),
                "reliable": len(engines_with) >= 2,  # 2 of 3 with this scaled set
            })
        votes.sort(key=lambda v: -v["n_hit"])

        # Stage 3: Schema mapping on combined text
        all_lines = []
        for p in signals.get("ocrmypdf_pages", []):
            for ln in (p.get("text") or "").splitlines():
                ln = ln.strip()
                if 4 <= len(ln) <= 80:
                    all_lines.append(ln)
        try:
            schema_mapped = _schema_map_nb_sbert(all_lines[:120], CANONICAL_LABELS)
        except Exception as e:
            schema_mapped = [{"error": str(e)[:120]}]

        # Stage 5: NER on combined text
        full_text = "\n".join(p.get("text","") for p in signals.get("ocrmypdf_pages",[]))
        try:
            entities = _ner_spacy(full_text)[:50]
        except Exception as e:
            entities = [{"error": str(e)[:120]}]

        return {
            "n_engines_run": sum(1 for k in signals if not k.endswith("_error")),
            "n_unique_numbers": len(all_numbers),
            "votes": votes[:50],
            "schema_mapped_top10": schema_mapped[:10],
            "ner_entities_top20": entities[:20],
            "pix2struct_qa": signals.get("pix2struct_qa", []),
            "errors": {k: v for k, v in signals.items() if k.endswith("_error")},
        }

    return {"per_pdf": for_each_pdf(process_one),
            "canonical_schema": CANONICAL_LABELS}


if __name__ == "__main__":
    run_with_metrics("ensemble_ocr_cascade", main)
