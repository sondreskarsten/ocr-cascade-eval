"""Path A — Tesseract custom traineddata for Norwegian årsregnskap typography.

Goal: drop the residual TSV miss rate by training tesseract's LSTM on synthetic
pages that match BRREG render typography (Times Roman 10pt, BRREG layout,
parens-negation, digit clusters with thousands-separator spaces).

Pipeline:
1. Render synthetic pages from real BRREG PDF text + structured Gemini extracts
2. Use tesstrain (https://github.com/tesseract-ocr/tesstrain) to fine-tune
   nor.traineddata via lstmtraining
3. Validate on v2 fixture against BRREG truth — measure tesseract_tsv recall
   improvement

Inputs:
  - 12,879 Gemini extractions at gs://sondre_brreg_data/raw/noter_extraction_2025/raw/
  - 12,879 source PDFs at gs://brreg-regnskap/regnskap/{orgnr}/aarsregnskap_{year}.pdf
  - tesstrain repo (cloned at runtime)
  - Pillow + freetype-py for typography rendering

Output:
  - gs://sondre_brreg_data/raw/tesseract_finetune/synthetic_corpus/
  - gs://sondre_brreg_data/raw/tesseract_finetune/checkpoints/nor_brreg.traineddata
  - gs://sondre_brreg_data/raw/tesseract_finetune/eval/v2_fixture_recall.json

Cost estimate: ~2 days wall (mostly synthetic data generation + training).
GPU: not strictly required for tesseract LSTM (CPU-trainable but slow);
recommend GPU for the synthetic data generation step (PIL rendering parallelism).
"""
import os
import re
import json
import random
from pathlib import Path

# === Stage 1: Build synthetic corpus from real Gemini extracts ===

# Each Gemini JSON has structured note titles + amount values.
# We render synthetic pages mimicking BRREG layout: label-left, value-right
# with numeric thousand-separator spacing.

LAYOUT_TEMPLATES = {
    "balanse_row": "{label:<55}{value_curr:>15}{value_prev:>15}",
    "resultatregnskap_row": "{label:<55}{value_curr:>15}{value_prev:>15}",
    "noter_lonn": "Lønn {value_curr:>15}{value_prev:>15}",
    "negative_paren": "{label:<55}({value_abs:>13}){value_prev:>15}",  # parens-negation
    "negative_minus": "{label:<55}-{value_abs:>14}{value_prev:>15}",
}


def format_norwegian_number(n):
    """Format integer with space as thousands-separator (Norwegian style)."""
    if n is None: return ""
    s = str(abs(int(n)))
    # Insert thin spaces every 3 digits from right
    s_rev = s[::-1]
    grouped = " ".join(s_rev[i:i+3] for i in range(0, len(s_rev), 3))[::-1]
    return ("-" + grouped) if n < 0 else grouped


def gen_balanse_page(rows, fontsize=24, page_width=2480, page_height=3508):
    """Render synthetic balanse page using PIL + Times Roman."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", fontsize)
    except OSError:
        font = ImageFont.load_default()
    
    margin_left = 200
    margin_top = 300
    line_h = fontsize + 14
    
    # Header
    draw.text((margin_left, margin_top - line_h * 2), "BALANSE - EIENDELER", font=font, fill="black")
    draw.text((1500, margin_top - line_h), "2024", font=font, fill="black")
    draw.text((1900, margin_top - line_h), "2023", font=font, fill="black")
    
    y = margin_top
    annotations = []  # (label, value_curr, value_prev, y, label_x, val_curr_x, val_prev_x)
    for row in rows:
        label = row["label"]
        v_curr = row["value_curr"]
        v_prev = row.get("value_prev")
        
        draw.text((margin_left, y), label, font=font, fill="black")
        v_curr_str = format_norwegian_number(v_curr) if v_curr else ""
        v_prev_str = format_norwegian_number(v_prev) if v_prev else ""
        # right-align numbers at fixed x
        v_curr_w = draw.textlength(v_curr_str, font=font)
        v_prev_w = draw.textlength(v_prev_str, font=font)
        draw.text((1500 + (200 - v_curr_w), y), v_curr_str, font=font, fill="black")
        draw.text((1900 + (200 - v_prev_w), y), v_prev_str, font=font, fill="black")
        
        annotations.append({
            "label": label,
            "value_curr": v_curr,
            "value_prev": v_prev,
            "y": y,
            "label_bbox": [margin_left, y, margin_left + draw.textlength(label, font=font), y + line_h],
        })
        y += line_h
    
    return img, annotations


def build_corpus_from_gemini(gemini_blob_iter, out_dir, max_pages=5000):
    """Stage 1: Pull Gemini extracts, render N synthetic pages."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
    
    pages_written = 0
    rng = random.Random(42)
    
    for blob in gemini_blob_iter:
        if pages_written >= max_pages: break
        try:
            d = json.loads(blob.download_as_bytes())
        except: continue
        
        # Synthesize a balanse-style row set from the noter (since Gemini doesn't
        # capture the resultat/balanse rows directly — those come from regnskap-cdc)
        notes = d.get("notes") or d.get("noter") or []
        rows = []
        for note in notes:
            title = (note.get("title") or "").strip()
            text = (note.get("full_text") or "").strip()
            # Find candidate (label, value) pairs in the text
            for m in re.finditer(r"^([A-ZÆØÅa-zæøå][^\n0-9]{3,40})\s+(-?\d{1,3}(?:[\s.]\d{3})+|-?\d+)", text, re.M):
                label = m.group(1).strip()
                val = int(re.sub(r"[^\d-]", "", m.group(2)))
                rows.append({"label": label, "value_curr": val,
                             "value_prev": int(val * rng.uniform(0.7, 1.3))})
                if len(rows) >= 20: break
            if len(rows) >= 20: break
        
        if len(rows) < 5: continue
        
        # Render
        try:
            img, annotations = gen_balanse_page(rows[:20])
        except Exception as e:
            print(f"  render failed: {e}")
            continue
        
        # Save
        page_id = f"page_{pages_written:06d}"
        img.save(out_dir / "images" / f"{page_id}.png")
        # Tesseract training expects .gt.txt — concatenated ground-truth text per line
        gt_lines = []
        for ann in annotations:
            line = f"{ann['label']} {format_norwegian_number(ann['value_curr'])}"
            if ann.get("value_prev"):
                line += f" {format_norwegian_number(ann['value_prev'])}"
            gt_lines.append(line)
        (out_dir / "ground_truth" / f"{page_id}.gt.txt").write_text(
            "\n".join(gt_lines), encoding="utf-8")
        # Per-line bbox annotations for tesstrain
        with open(out_dir / "ground_truth" / f"{page_id}.box", "w") as f:
            for ann in annotations:
                # box format: char x0 y0 x1 y1 page_n
                # tesstrain consumes line-level boxes
                bbox = ann["label_bbox"]
                f.write(f"{ann['label']} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} 0\n")
        
        pages_written += 1
        if pages_written % 100 == 0:
            print(f"  rendered {pages_written} pages...")
    
    print(f"\nCorpus complete: {pages_written} pages in {out_dir}")
    return pages_written


# === Stage 2: lstmtraining ===

LSTMTRAINING_BASH = """#!/bin/bash
# Run on a machine with tesseract-ocr-dev + tesstrain installed
# Estimated time: ~4-12 hours on GPU, ~24-72 hours on 8-core CPU
set -euo pipefail

CORPUS_DIR=${1:-/data/synthetic_corpus}
OUT_DIR=${2:-/data/finetune_out}
START_MODEL=${3:-/usr/share/tesseract-ocr/4.00/tessdata/nor.traineddata}

# Clone tesstrain if not present
if [ ! -d /opt/tesstrain ]; then
  git clone https://github.com/tesseract-ocr/tesstrain.git /opt/tesstrain
fi

cd /opt/tesstrain

# Run training
make training MODEL_NAME=nor_brreg \\
  START_MODEL=nor \\
  TESSDATA_REPO=/usr/share/tesseract-ocr/4.00/tessdata \\
  GROUND_TRUTH_DIR=$CORPUS_DIR/ground_truth \\
  OUTPUT_DIR=$OUT_DIR \\
  MAX_ITERATIONS=50000 \\
  LANG_TYPE=Indo-European \\
  PSM=6

# Final model lands at $OUT_DIR/nor_brreg.traineddata
"""


# === Stage 3: Eval against v2 fixture ===

def eval_finetuned_on_v2(traineddata_path, fixture_dir):
    """Run new tesseract model on v2 fixture pages, measure recall vs BRREG truth."""
    import pytesseract
    # Override tessdata location
    os.environ["TESSDATA_PREFIX"] = os.path.dirname(traineddata_path)
    
    from google.cloud import storage
    cli = storage.Client()
    
    truth = {}
    for blob in cli.list_blobs("sondre_brreg_data",
        prefix="raw/ocr_eval_v2_10pdfs_300dpi/audit/brreg_ground_truth/"):
        if blob.name.endswith(".json"):
            d = json.loads(blob.download_as_bytes())
            nums = set()
            for k, v in d.get("key_metrics", {}).items():
                if isinstance(v, (int, float)) and v != 0:
                    nums.add(int(round(v)))
            truth[d["orgnr"]] = nums
    
    total_hit = 0
    total_truth = 0
    per_pdf = {}
    for orgnr, truth_nums in truth.items():
        page_imgs = sorted(Path(fixture_dir).glob(f"{orgnr}_p-*.png"))
        all_extracted = set()
        for img_path in page_imgs:
            text = pytesseract.image_to_string(str(img_path),
                lang="nor_brreg", config="--psm 6")
            for m in re.finditer(r"-?\d{1,3}(?:[\s.]\d{3})+|-?\d+", text):
                try:
                    val = int(re.sub(r"[^\d-]", "", m.group()))
                    if abs(val) >= 10: all_extracted.add(val)
                except: pass
        
        hits = sum(1 for v in truth_nums if v in all_extracted or -v in all_extracted)
        per_pdf[orgnr] = {"hits": hits, "truth_n": len(truth_nums),
                          "recall": hits / len(truth_nums) if truth_nums else None}
        total_hit += hits
        total_truth += len(truth_nums)
    
    return {
        "model": "nor_brreg.traineddata",
        "v2_recall_overall": total_hit / total_truth if total_truth else None,
        "v2_total_hit": total_hit,
        "v2_total_truth": total_truth,
        "per_pdf": per_pdf,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build_corpus", "train", "eval"], required=True)
    parser.add_argument("--out_dir", default="/tmp/tesseract_finetune")
    parser.add_argument("--traineddata", default=None)
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--max_pages", type=int, default=5000)
    args = parser.parse_args()
    
    if args.stage == "build_corpus":
        from google.cloud import storage
        cli = storage.Client()
        blobs = (b for b in cli.list_blobs("sondre_brreg_data",
            prefix="raw/noter_extraction_2025/raw/") if b.name.endswith(".json"))
        n = build_corpus_from_gemini(blobs, args.out_dir, max_pages=args.max_pages)
        print(f"Built {n} synthetic pages at {args.out_dir}")
    elif args.stage == "train":
        print("Run: bash finetune/tesseract/lstmtrain.sh")
        print("Bash script:")
        print(LSTMTRAINING_BASH)
    elif args.stage == "eval":
        if not args.traineddata or not args.fixture:
            raise SystemExit("--traineddata and --fixture required")
        result = eval_finetuned_on_v2(args.traineddata, args.fixture)
        print(json.dumps(result, indent=2, ensure_ascii=False))
