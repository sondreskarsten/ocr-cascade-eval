# Additional OCR signals — beyond text-mode

## Discovered during cascade voting calibration

When tesseract's `image_to_string` text-mode missed 6 numeric values on the v2 10-PDF fixture, raster review showed the page contained those numbers clearly. The post-processed text mode collapses adjacent number-tokens incorrectly, but the **word-level TSV stream** (`pytesseract.image_to_data`) preserves them with high confidence.

## Empirical test (v2 fixture, 3 problem PDFs)

Re-ran tesseract TSV on the 3 PDFs where text-mode missed values:

| orgnr | text-mode hits | TSV hits | improvement |
|---|---|---|---|
| 820746112 | 7/10 | 9/10 | +2 (sum_eiendeler 16 164 967 ✓, sum_egenkapital_gjeld ✓) |
| 818751192 | 2/5 | 5/5 | +3 (full balanse recovery) |
| 820246012 | 8/9 | 8/9 | (1 miss = "97" — single 2-digit number) |

Total: **22/24 = 92% recall** on previously-failing cases. The 2 remaining misses are visible in TSV but require relaxed token filters (digits ≥ 100 → digits ≥ 10).

## Why text-mode loses values that TSV preserves

Tesseract default text output applies line-merging heuristics that fail when:
- A row has very wide column spacing (causes "line_num" boundary errors).
- A balanse-style table has labels on the left and values aligned to fixed right columns separated by 200+ pixels of whitespace.
- A digit cluster like `"1 135 691"` gets split by spurious column-detection.

The TSV output sidesteps this by exposing per-token bbox + confidence + line_num. We can re-cluster tokens by row-y and column-x ourselves, with our own merging heuristic:

```python
import pytesseract, re

def tesseract_tsv_numbers(img_path, lang="nor"):
    data = pytesseract.image_to_data(
        img_path, lang=lang, output_type=pytesseract.Output.DICT)
    by_line = {}
    for i, w in enumerate(data["text"]):
        w = (w or "").strip()
        if not w: continue
        if data["conf"][i] < 30: continue
        line_id = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        by_line.setdefault(line_id, []).append({
            "text": w, "left": data["left"][i],
            "conf": data["conf"][i],
        })
    found = set()
    for tokens in by_line.values():
        tokens.sort(key=lambda t: t["left"])
        # cluster adjacent number-only tokens into integers
        i = 0
        while i < len(tokens):
            t = tokens[i]["text"]
            if not re.match(r"^-?\d+$", t):
                i += 1; continue
            sign = "-" if t.startswith("-") else ""
            digits = re.sub(r"[^\d]", "", t)
            j = i + 1
            while j < len(tokens):
                tn = tokens[j]["text"]
                if (re.match(r"^\d{1,3}$", tn) and
                    tokens[j]["left"] - tokens[j-1]["left"] < 250):
                    digits += tn; j += 1
                else: break
            try:
                v = int(sign + digits)
                if abs(v) >= 10: found.add(v)
            except: pass
            i = max(j, i + 1)
    return found
```

## Other latent signals worth wiring up

### 1. **paddleocr per-line confidence**
Already in result payload as `lines[i].conf`. Empirical pattern: paddleocr labels rows with conf < 0.85 are 3-4× more likely to contain digit substitutions. **Action**: flag low-conf rows for cross-engine vote validation.

### 2. **doctr line-level position**
doctr returns word-level bbox — same merging trick as tesseract TSV applies. Currently the runner concatenates words with " " which loses thousands-separator structure. **Action**: same per-row token clustering with bbox-aware joining.

### 3. **ocrmypdf hOCR output**
ocrmypdf can emit hOCR (HTML with embedded bbox/conf). We currently use only the text extracted by pdfplumber from the OCR'd PDF. **Action**: parse hOCR directly, get word-level confidence as another signal.

### 4. **Cross-engine x-alignment voting**
Each engine reports rough x-coordinates for tokens. If all engines that *do* find a number agree on x ≈ 1495-1593 (typical "current year" column), flag any row where label-x is right but value-x is missing as "column-drop suspected".

## Possible fine-tuning paths

### A. Tesseract custom traineddata for Norwegian financial vocab + digits
Tesseract's `nor.traineddata` is general-purpose. We could:
1. Generate synthetic training pages from Norwegian årsregnskap typography (Times Roman 10pt, BRREG layout).
2. Fine-tune via Tesseract's `lstmtraining` toolchain on ~10K rendered samples.
3. Specifically optimize: digit-cluster recognition (`1 135 691` as one entity), Norwegian number prefixes ("kr", "NOK"), parenthesis-negation `(123)` → `-123`.

Cost: ~2 days setup + GPU training. Expected gain: drops the residual 8% TSV miss rate.

### B. Fine-tune a small VLM (Donut/Pix2Struct-base) on Norwegian regnskap
Donut-base has the right architecture (image → structured JSON) but its CORD-v2 fine-tune is for Korean restaurant receipts. Fine-tuning steps:
1. Use the 12,879 Gemini extractions as silver labels.
2. Render each PDF as page images, train Donut to output `{"resultatregnskap": [{label, amount}], ...}`.
3. Validate held-out set against BRREG live API.

Cost: ~1 week + GPU. Expected gain: a model that produces structured output without the OCR→regex pipeline. Could replace Gemini at inference time.

### C. Train a row-level classifier on TSV features
Per-row features from tesseract TSV: `[avg_conf, n_tokens, has_digit_cluster, x_span, label_distance]`. Use the v2 BRREG truth labels as supervised targets — does this row contain a real nøkkeltall? Trained on 10 PDFs is overfit-prone, but on 12K PDFs it's a usable supervised signal. Could output a confidence score per extraction.

## Recommendation

For the next iteration of the ensemble runner, add:
1. **`tesseract_tsv` runner** (new): TSV-based numeric extraction with row-aware token clustering. **Does not replace text mode, runs alongside.** Its votes count as an additional engine in the cascade.
2. **`paddleocr_conf` filter**: in voting, weight votes by per-engine confidence (down-weight paddleocr lines with conf < 0.85).
3. **`doctr_bbox` runner**: same row-clustering approach on doctr word output.
4. **Vote threshold = 4/8** (was 4/6) — re-run the audit, see if any disagreements remain.

Fine-tuning paths (A/B/C above) are bigger investments to defer until the baseline ensemble is stable on 12k.
