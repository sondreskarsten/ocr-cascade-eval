# Multi-DGP Audit Pattern

A reusable pattern for evaluating any system where N independent mechanisms observe the same underlying state. The OCR cascade in this repo is the canonical instance.

---

## Premise

Naive empiricism (Sondre Skarsten / Registrum platform):
> Each parquet row is an immutable observation at its own timestamp. No retroactive truth, no cross-source schema unification at write time — unification is query-time only, preserving lead-lag signal.

Apply this to **measurement**: when N mechanisms read the same underlying state, each is its own DGP-aspect. Store each independently, vote at query time, calibrate disagreements via direct inspection.

---

## When to use this pattern

Whenever you have:
1. **A canonical truth** that exists independently of the observation mechanisms (e.g. BRREG live API for nøkkeltall, a Brreg bulk dump for org-state, a chart of accounts for line-item mapping).
2. **N independent mechanisms** that produce observations of that truth (different OCR engines, different APIs, different scrapers, different annotators).
3. **The mechanisms have known/expected disagreement modes** (OCR has column-drop, scrapers have rate limits, APIs have schema drift, annotators have tagging bias).

If you only have one mechanism, this pattern is overkill — just measure recall against truth. If you have no canonical truth, this is inter-rater agreement, not multi-DGP audit.

---

## Five-step procedure

### 1. Inventory mechanisms as separate DGPs

Each mechanism is its own ledger. Don't unify schemas at write time.

```
gs://.../raw/cascade_eval_FIXTURE/results/{mechanism_name}.json
```

In the OCR case:
- 6 text-mode engines: tesseract, paddleocr, ocrmypdf, doctr, easyocr, nougat
- 4 word-level signals: tesseract_tsv, doctr_bbox, ocrmypdf_hocr, paddleocr_conf
- 1 visual extractor: pix2struct

10 separate result files. Each is admissible at its own timestamp.

### 2. Define the truth source

The truth must be:
- **Independent** of the mechanisms being measured (otherwise you're measuring how well X agrees with X)
- **Stable** at the time of measurement (no retroactive truth)
- **At the same LUAS** as the disagreements you want to detect

In OCR: BRREG live API `/regnskapsregisteret/regnskap/{orgnr}` returns key_metrics. 10 PDFs × ~10 metrics each = 100 truth values.

```python
truth = {orgnr: {label: int_value, ...} for orgnr in fixture}
```

### 3. Vote per truth value, with normalization at compare time

```python
DIA = {'æ':'a','ø':'o','å':'a','Æ':'A','Ø':'O','Å':'A'}
def normalize(s): return ''.join(DIA.get(c, c) for c in s).lower()

for orgnr in fixture:
    for label, truth_value in truth[orgnr].items():
        engines_hit = [
            engine for engine in MECHANISMS
            if value_in_observation(engine, orgnr, truth_value)
        ]
        consensus = len(engines_hit)
        # consensus >= 7 -> reliable
        # consensus == 0 -> universal miss (investigate)
        # 0 < consensus < N -> disagreement (raster review)
```

**Normalization at compare time, not write time.** This is the diacritic finding from this repo: doctr strips æ→a but the recall jumps from 42% → 67% when you normalize *both sides* of the compare. Don't store stripped output — store raw, normalize when comparing.

### 4. Categorize the consensus distribution

| consensus | interpretation | action |
|---|---|---|
| N/N (unanimous) | extremely high confidence | accept |
| ≥ ⌈N×0.7⌉ | reliable consensus | accept with weight |
| 4 ≤ x < ⌈N×0.7⌉ | mixed, disputed | flag for raster review |
| 1 ≤ x ≤ 3 | minority observation | likely engine error |
| 0 (universal miss) | every engine missed it | data error or systemic bias |

In the OCR case (N=10): 81 unanimous, 99 reliable, 19 disagreements, 0 universal misses.

### 5. Raster calibration on disagreements

For each disagreement, render the underlying primitive (page, document, record) at high fidelity and visually verify which mechanisms are right.

```python
for case in disagreement_cases:
    raster = render_at_high_dpi(case.pdf, case.page, case.bbox)
    save_to_gcs(raster, case.crop_path)
    # Manual review: which engines were right?
```

The 14 disagreements in the v2 fixture were all explained by specific failure modes (column-drop, no_extraction, OCR_substitution, word_fragmentation, leading_digit_drop) — every miss visually confirmed.

---

## What this pattern produces

Per truth value:
- `consensus_count` — how many mechanisms agree
- `engines_hit` / `engines_missing` — explicit list, useful for engine-level diagnostics
- `reliable: bool` — derived from consensus threshold
- `failure_mode` (post-calibration) — categorized per mechanism

Per mechanism:
- `miss_rate` — fraction of truth values missed
- `failure_mode_distribution` — what kind of mistakes does this mechanism make?
- `complementarity` — when this mechanism misses, do others catch it? (for ensemble design)

Per fixture:
- `unanimous_rate` — how often do all mechanisms agree?
- `universal_miss_rate` — how often does every mechanism miss? (true blind spots)
- `mean_consensus` — average of consensus_count

---

## Generalizations beyond OCR

### N APIs reading the same firm

Brreg / Finanstilsynet / Skatteetaten / Mattilsynet may all have records for the same orgnr. Each is a DGP-aspect:
- Brreg: legal entity registration
- Finanstilsynet: financial license status
- Skatteetaten: tax assessment
- Mattilsynet: food/health inspection

Truth: the firm's actual operating state. Vote on existence + activity status across sources, calibrate disagreements by checking the underlying registry filings.

### N scrapers parsing the same kunngjøring

Multiple scrapers (kunngjoring-collector, doffin-collector, downstream Bayesian classifiers) may all attempt to detect a konkurs event for the same orgnr. Each is a DGP-aspect:
- Production CDC pipeline
- Newspaper/NB.no media classifier  
- DN print archive
- DNB internal D&B feed

Truth: the konkurs actually happened (or didn't) per legal filing. Vote on event detection across sources. Disagreement → raster the underlying source documents.

### N annotators labeling the same firm event

Multiple human annotators classify the same kunngjøring ktype. Each is a DGP-aspect (an annotator). Truth: ground-truth label assigned by an authoritative reviewer. Vote on label assignment, calibrate via direct review of the announcement text.

### N model versions on the same dataset

Run model_v1, model_v2, model_v3 on the same dataset. Each is a DGP-aspect. Truth: held-out validation labels. Vote on prediction agreement, calibrate via direct review of disagreement cases. This is essentially how you decide which model version to ship.

---

## Anti-patterns to avoid

1. **Cross-source schema unification at write time.** If you collapse "ocrmypdf says 16164967" and "tesseract says 16" into a single "extracted_value" column at write time, you've lost the lead-lag signal. Store each separately. Unify at query time.

2. **Treating one mechanism as truth.** If you treat Gemini as truth and measure ocrmypdf against Gemini, you're measuring how well ocrmypdf agrees with Gemini, not how well it extracts the underlying number. Always anchor against an independent truth source.

3. **Unanimous = correct.** Unanimous consensus is high confidence, not certainty. Check whether all mechanisms share a common failure mode (e.g. all OCR engines miss numbers in cells with rotated text). Universal miss = real data state to investigate.

4. **Raw recall as the metric.** Two mechanisms can have identical recall but very different complementarity. Always look at *which* truth values each mechanism catches/misses, not just the count. The 4-engine word-level signal added redundancy on the same values text-mode caught — net negative.

5. **Stripping observations to fit comparisons.** If doctr strips diacritics, don't store stripped text. Store raw, normalize at compare time. Other downstream uses (NER, semantic search) need the diacritics.

---

## Reference implementation

This repo's `pipelines/ensemble_ocr_cascade.py` + `pipelines/xalign_vote.py` + `audit/cascade/v2_10signals/` is the reference.

The audit JSON schema is reusable — same structure for any multi-DGP evaluation:

```json
{
  "_meta": {
    "fixture": "...",
    "engines": ["...", "..."],
    "truth_source": "...",
    "n_truth_values": 100
  },
  "voting_per_orgnr_label": {
    "{orgnr}": {
      "{label}": {
        "truth_value": 16164967,
        "engines_hit": ["paddleocr", "ocrmypdf", ...],
        "engines_missing": ["tesseract", "tesseract_tsv"],
        "n_hit": 8,
        "unanimous": false,
        "reliable": true
      }
    }
  },
  "consensus_distribution": {"10": 81, "9": 18, "6": 1},
  "miss_by_engine": {"tesseract": 6, "tesseract_tsv": 1, ...},
  "disagreement_cases": [...]
}
```
