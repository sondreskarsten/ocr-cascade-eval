# ocr-cascade-eval

Empirical evaluation of every OCR / document-AI / embedding / table-QA / financial NLP / Norwegian LLM tool referenced — positively or negatively — in the "Norwegian årsregnskap parser" report. The article calls most of them irrelevant for the schema-mapping task; we treat that as an empirical claim and check it.

## Test fixture

`gs://sondre_brreg_data/raw/ocr_eval_2026_05_05/fixture/`

- `test.pdf` — FARBOSS AS årsregnskap 2024 (orgnr 814 747 352, 10 pages, image-only scan)
- `test_ocr.pdf` — same with `ocrmypdf -l nor` text overlay
- `pages_p02.png` — resultatregnskap (income statement)
- `pages_p06.png` — noter table page
- `tesseract_input.json` — words+boxes from `tesseract -l nor` for layout-aware models
- `table.csv` — small structured table for table-QA models
- `samples.json` — Norwegian financial sentences, label, NER text, candidate label set

## Models covered

30 jobs across 5 base images. One file per model under `models/`.

| Image | Models |
|---|---|
| `ocr-eval-paddle` | paddleocr |
| `ocr-eval-table` | tesseract, camelot, tabula, ocrmypdf |
| `ocr-eval-hf` | donut, layoutlmv3, udop, lilt, bge_m3, multilingual_e5, nb_sbert, bge_reranker_v2, finbert_prosus, finbert_hkust, finbert2, fingpt, finma, byt5, gliner, gliner2, gliclass, setfit, tapas, tapex, omnitab |
| `ocr-eval-llm` | normistral_11b, norwai_magistral_24b (GGUF Q4_K_M via llama.cpp) |
| `ocr-eval-xbrl` | arelle, pyesef |

Mentioned but not shipped:
- BloombergGPT — closed source; cannot be loaded
- TableLlama — 13B+ on Llama-2; doesn't fit 32 Gi Cloud Run
- jina-embeddings-v3 — CC-BY-NC; license excluded by article
- DSPy / Magneto / ReMatch / SCHEMORA — frameworks/papers, not single-model checkpoints
- RapidFuzz / Pydantic / Pandera / Outlines / Instructor / Marvin / XGrammar — libraries, not models

## Output

Each job writes `gs://sondre_brreg_data/raw/ocr_eval_2026_05_05/results/{model}.json` with:

```
{
  "model": "...",
  "status": "ok" | "error",
  "wall_s": float,
  "rss_mb_start": float,
  "rss_mb_end": float,
  "rss_mb_delta": float,
  "error_type": "...",
  "error_msg": "...",
  ...model-specific output...
}
```

## Workflow

```bash
# 1. Build all 5 images via Cloud Build (parallel)
gcloud builds submit --config=cloudbuild.yaml --project=sondreskarsten-d7d14 .

# 2. Deploy / update all 30 Cloud Run Jobs
GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json python3 scripts/deploy_jobs.py

# 3. Trigger all jobs in parallel (returns immediately; jobs run on GCP)
GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json python3 scripts/trigger_jobs.py

# 4. Check status (poll any time)
python3 scripts/check_status.py

# 5. Aggregate results when done
python3 scripts/aggregate_results.py
```

## Architecture choices

- One image per dependency-cluster, not one per model — avoids 30 separate builds
- Cloud Run Jobs (not Services) — no HTTP, fire-and-forget batch
- Scratch only writes to GCS — no shared volume, no orchestration server
- Models download checkpoints at runtime from Hugging Face — image stays small (~3 GB), but cold-start adds 30 s – 5 min per job
- Each job writes its own result file. Aggregation is read-only over GCS

## Resource sizing

- Default 16 Gi / 4 CPU
- 8 Gi for small models (Tesseract baseline, FinBERT-tone, NB-SBERT, ByT5)
- 32 Gi for LoRA-on-Llama2-7B (FinGPT, FinMA) and NorwAI-Magistral-24B-Q4
- Cloud Run hard cap: 32 Gi memory, 8 vCPU, 24 h timeout
