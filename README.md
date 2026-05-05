# ocr-cascade-eval

Multi-signal OCR cascade for Norwegian årsregnskap PDFs. **77 OCR / document-AI / vision-LM / NLP models evaluated as independent observation mechanisms (DGPs) of the same underlying document, with cross-engine voting + raster calibration against BRREG live API truth.**

> _If you're looking for the public benchmark notebook for the full 12,879-PDF sweep, see [`sondreskarsten/norwegian-ocr-benchmark`](https://github.com/sondreskarsten/norwegian-ocr-benchmark)._

---

## Why this exists

Norwegian årsregnskap PDFs come in heterogeneous layouts (small-firm vs konsern, BRREG-rendered vs scanned, table-heavy balanse vs narrative noter). No single OCR engine handles all of them well. This repo treats each engine as a separate **DGP-aspect** of the same document — gather independent observations, vote on numeric values, calibrate disagreements by inspecting the underlying raster directly. The framing matches the broader Registrum platform's naive-empiricism principle: each row is an immutable observation at its own timestamp; meaning is unified at query time, not write time.

The fixture (`gs://sondre_brreg_data/raw/ocr_eval_v2_10pdfs_300dpi/`) is a **canonical multi-DGP example** for the platform — same pattern generalizes to any case where N independent mechanisms read the same underlying state.

---

## Cascade architecture

```
                        ┌───────────────────────────────┐
                        │   PDF (300 DPI page rasters)  │
                        └───────────────┬───────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
   ┌────▼────┐                ┌────────▼────────┐                ┌─────▼─────┐
   │TEXT (6) │                │ WORD-LEVEL (4)  │                │VISUAL (1) │
   │tesseract│                │ tesseract_tsv   │                │pix2struct │
   │paddleocr│                │ doctr_bbox      │                │docvqa-base│
   │ocrmypdf │                │ ocrmypdf_hocr   │                └───────────┘
   │doctr    │                │ paddleocr_conf  │
   │easyocr  │                └─────────────────┘
   │nougat   │
   └────┬────┘
        │
   ┌────▼─────────────────────────────────────────────────────┐
   │ Stage 2: Numeric voting per truth-value                  │
   │  - normalize() applied at compare time (æ→a, ø→o, å→a)   │
   │  - thousands-separator-tolerant integer matching         │
   │  - consensus_count = engines hitting truth value         │
   └────┬─────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────────────┐
   │ Stage 3: Schema mapping (nb_sbert)                       │
   │  - line → canonical nøkkeltall label                     │
   │  - Norwegian-trained, threshold-friendly spread (0.75)   │
   └────┬─────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────────────┐
   │ Stage 4: Fill-mask normalization (nb_bert)               │
   └────┬─────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────────────┐
   │ Stage 5: NER on noter (spacy_nb / gliner2)               │
   └────┬─────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────────────┐
   │ Stage 6: X-alignment column-drop detector                │
   │  - flag rows where label-x ✓ but value-x ✗               │
   └────┬─────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────────────┐
   │ Stage 7: Raster calibration                              │
   │  - render disagreements at 300 DPI, verify visually      │
   └──────────────────────────────────────────────────────────┘
```

---

## Empirical results — v2 fixture (10 PDFs, true 300 DPI)

| metric | value |
|---|---:|
| Total truth values (BRREG live API key_metrics) | 100 |
| **Unanimous (10/10 engines agree)** | **81** |
| **Reliable (≥7/10)** | **99** |
| Disagreement (1 ≤ n_hit ≤ 9) | 19 |
| Universal miss (0/10) | 0 |

### Per-engine reliability

| engine | type | misses (out of 100) | notes |
|---|---|---:|---|
| paddleocr | text | **0** | perfect on this fixture |
| ocrmypdf | text | **0** | 100% recall, 15s/10 PDFs (200× faster than paddleocr) |
| doctr | text | **0** | with normalize-on-compare; raw=42% match (diacritic-strip artifact) |
| **tesseract_tsv** | word | 1 | recovers 5 of 6 tesseract text-mode misses |
| ocrmypdf_hocr | word | 1 | row-cluster matches text mode |
| doctr_bbox | word | 2 | row-clustering brittle |
| nougat | text | 4 | designed for narrative, not tables |
| easyocr | text | 4 | word fragmentation + occasional hallucinations |
| paddleocr_conf | word | 4 | row-clustering brittle |
| tesseract | text | 6 | column-drop on balanse layouts (fixed by tesseract_tsv) |

### Word-level vs text-mode pairings

Same OCR engine, different post-processing. Row-cluster aggregation on per-token bbox output **does not always help**:

| pair | text misses | word misses | net |
|---|---:|---:|---:|
| **tesseract → tesseract_tsv** | 6 | 1 | **+5** ✓ |
| ocrmypdf → ocrmypdf_hocr | 0 | 1 | -1 |
| doctr → doctr_bbox | 0 | 2 | -2 |
| paddleocr → paddleocr_conf | 0 | 4 | -4 |

Only `tesseract_tsv` is a net positive. The others' row-clustering heuristic introduces its own column-merge errors that text-mode avoids.

### Production verdict (7-voter cascade)

| signal | role |
|---|---|
| ocrmypdf | primary OCR (fastest at 100% recall) |
| paddleocr | redundancy + confidence baseline |
| doctr | redundancy + Latin-only normalization check |
| tesseract | text-mode baseline |
| **tesseract_tsv** | **column-drop recovery** (only word-level signal worth keeping) |
| easyocr | fragmentation sentry (low weight in vote) |
| nougat | narrative noter check (text only, not balanse) |

**Drop from production**: `doctr_bbox`, `ocrmypdf_hocr`, `paddleocr_conf` — useful as benchmark sentries, but their row-clustering errors hurt cascade voting.

### Companion LLM stack (per-stage best)

- **Schema mapping** (stage 3): `nb_sbert` (avg 0.57, spread 0.75 — best for thresholding)
- **Fill-mask** (stage 4): `nb_bert` (predicts "resultat" with logit 16.5 on "Selskapets [MASK] for 2024 var positivt")
- **NER on noter** (stage 5): `gliner2` (most coverage) or `spacy_nb` (most balanced)
- **Visual fallback** (no OCR): `pix2struct-docvqa-base` (90% driftsresultat / 70% årsresultat)

---

## Directory layout

```
ocr-cascade-eval/
├── runners/                          # one Python file per OCR engine / signal (77 total)
│   ├── tesseract.py / paddleocr.py / ocrmypdf.py / doctr.py / easyocr.py / nougat.py
│   ├── tesseract_tsv.py              # word-level signal — NET +5 misses recovered
│   ├── doctr_bbox.py                 # word-level signal
│   ├── ocrmypdf_hocr.py              # word-level signal
│   ├── paddleocr_conf.py             # per-line confidence
│   ├── pix2struct.py                 # visual extraction (no OCR step)
│   ├── nb_sbert.py / nb_bert.py / spacy_nb.py / gliner2.py     # NLP stages
│   └── ...60+ more LLMs / embeddings / NERs / table-extractors
├── pipelines/
│   ├── ensemble_ocr_cascade.py       # full 7-stage cascade as a runner
│   ├── xalign_vote.py                # cross-engine column-drop detector
│   ├── SIGNALS.md                    # writeup of all word-level signals + diacritic finding
│   └── __init__.py
├── images/                           # 6 Dockerfiles (table, hf, llm, paddle, xbrl, supervisor)
├── jobs.json                         # 77 Cloud Run Job definitions
├── shared.py                         # fixture loader, GCS write, run_with_metrics
├── scripts/
│   ├── build_images.py               # parallel Cloud Build all images
│   ├── deploy_jobs.py                # create/patch 77 CRJs
│   ├── trigger_jobs.py               # fire all jobs
│   ├── check_status.py               # poll progress
│   ├── supervisor.py                 # autonomous retry, runs every 5min via Scheduler
│   └── aggregate_results.py
└── notebooks/
    └── (deferred to norwegian-ocr-benchmark for the public Colab)
```

---

## Workflow

```bash
# 1. Build all base images (parallel via Cloud Build)
GOOGLE_APPLICATION_CREDENTIALS=key.json python3 scripts/build_images.py

# 2. Deploy / update all 77 Cloud Run Jobs
GOOGLE_APPLICATION_CREDENTIALS=key.json python3 scripts/deploy_jobs.py

# 3. Trigger jobs (fire-and-forget; Cloud Scheduler supervisor handles retries)
GOOGLE_APPLICATION_CREDENTIALS=key.json python3 scripts/trigger_jobs.py

# 4. Audit
GOOGLE_APPLICATION_CREDENTIALS=key.json python3 scripts/aggregate_results.py
```

The supervisor (`scripts/supervisor.py`) runs every 5 min via Cloud Scheduler. It checks per-engine result files, retries failed runs up to 3×, and writes a snapshot to `gs://sondre_brreg_data/raw/ocr_eval_v2_10pdfs_300dpi/supervisor/` for monitoring.

---

## Output paths

```
gs://sondre_brreg_data/raw/ocr_eval_v2_10pdfs_300dpi/
├── fixture/
│   ├── pdfs/{orgnr}.pdf                    # source PDF
│   ├── pdfs_ocr/{orgnr}.pdf                # ocrmypdf-augmented (text layer added)
│   ├── pages/{orgnr}_p-NN.png              # per-page raster at 300 DPI
│   └── pdfs_meta.json                      # full_text, page_words (bbox), page_size
├── results/
│   └── {engine}.json                       # per-engine output, 77 files
├── audit/
│   ├── brreg_ground_truth/{orgnr}.json     # BRREG live API key_metrics
│   ├── gemini_ground_truth/{orgnr}.json    # Gemini 2.5 Flash full extraction
│   ├── full_v2_audit.{json,md}             # all 77 engines vs truth
│   ├── ranked_v2_audit.json                # OCR-focused ranking
│   ├── llm_audit.{json,md}                 # LLM categories (DocVQA, embeddings, classifiers)
│   └── cascade/
│       ├── voting.{json,md}                # 6-engine cascade
│       ├── calibrated_voting.md            # 14 disagreements visually verified
│       ├── noter_coherence.{json,md}       # noter prose quality per engine
│       ├── noter_verdict.md                # qualitative noter recommendation
│       └── v2_10signals/
│           ├── voting.{json,md}            # 10-signal cascade
│           ├── xalign_report.json          # x-alignment column-drop detector
│           └── SUMMARY.md                  # final 10-signal verdict
└── supervisor/                             # autonomous retry snapshots
```

---

## Multi-DGP framing

This evaluation is structured as a **multi-DGP audit pattern** that generalizes beyond OCR:

> Whenever N independent mechanisms observe the same underlying state, store each as its own ledger with its own LUAS, vote at query time, calibrate disagreements via direct inspection of the underlying primitive.

Examples elsewhere in the platform:
- N different APIs / bulk loaders reading the same Brreg dump → cross-source voting per orgnr
- N independent kunngjøring scrapers → consensus on event detection  
- N annotators labeling the same firm event → inter-annotator agreement at LUAS level

The OCR cascade is the most concrete instance: 10 engines, 100 truth values, 81 unanimous, 19 disagreements all traced to specific OCR-engine failure modes confirmed by raster inspection.

---

## Companion repos

- [`sondreskarsten/norwegian-ocr-benchmark`](https://github.com/sondreskarsten/norwegian-ocr-benchmark) — Colab notebook for the public 12,879-PDF benchmark, uses Colab built-in auth (no service-account key)

## Empirical claims

Every result in this README is reproducible from `gs://sondre_brreg_data/raw/ocr_eval_v2_10pdfs_300dpi/audit/cascade/v2_10signals/SUMMARY.md` and the underlying voting JSON.
