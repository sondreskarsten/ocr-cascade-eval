# Path B: Pix2Struct fine-tune on 12,879 Gemini extractions

Replace the OCR → schema mapping → fill-mask pipeline with a single vision-LM that outputs structured JSON directly from page rasters.

## Goal

`pix2struct-docvqa-base` already hits 90% driftsresultat / 70% årsresultat on the v2 fixture **without fine-tuning** (per `audit/cascade/v2_10signals/llm_audit.md`). With fine-tuning on 12,879 Gemini-extracted JSONs as silver labels, we should reach Gemini-equivalent quality at inference time and could replace the Gemini API call in the noter pipeline.

## Why pix2struct over donut

- pix2struct already shows non-trivial out-of-the-box performance on Norwegian årsregnskap. Donut requires Norwegian tokenization changes.
- pix2struct uses a screenshot-style ViT encoder that handles A4 pages at 300 DPI natively (max_patches=2048).
- pix2struct's text decoder is general-purpose; we just fine-tune it to emit structured JSON.

## Pipeline

```bash
# Stage 1: Build training JSONL from Gemini extractions
python pix2struct_finetune.py --stage build_jsonl \
    --out_path gs://sondre_brreg_data/raw/pix2struct_finetune/train.jsonl \
    --max_pdfs 11000

# Stage 2: Train on Colab Pro+ A100 80GB
# Open finetune_colab.ipynb, mount GCS, run all cells
# Saves checkpoints per epoch to gs://sondre_brreg_data/raw/pix2struct_finetune/checkpoints/

# Stage 3: Eval on v2 fixture against BRREG truth
python pix2struct_finetune.py --stage eval \
    --model_dir gs://sondre_brreg_data/raw/pix2struct_finetune/checkpoints/epoch_2 \
    --fixture gs://sondre_brreg_data/raw/ocr_eval_v2_10pdfs_300dpi/fixture/pages
```

## Data split

| split | n_PDFs | use |
|---|---:|---|
| train | 11,000 | gradient updates |
| validation | 1,000 | per-epoch loss tracking, early stopping |
| held-out test | 879 | touched only after final model is locked |
| v2 fixture | 10 | continuous benchmark, BRREG truth, raster-verified |

## Hyperparameters

- model: `google/pix2struct-base` or `google/pix2struct-docvqa-base` (warm-start)
- lr: 5e-5
- batch_size: 4 (A100 80GB)
- epochs: 3
- max_input_resolution: 4096×4096 (covers A4 at 300 DPI)
- max_target_tokens: 2048
- mixed precision: bf16

## Cost

- Data prep: 1 day (12,879 PDFs × ~5 pages = 60K samples, ~200 GB at 300 DPI)
- Training: 4-5 days on A100 80GB Colab Pro+ (3 epochs × ~1 sample/sec)
- Eval: 1 day (held-out + v2)

Total: ~1 week wall.

## Decision criteria

Promote pix2struct_brreg to production noter pipeline only if:
1. v2 recall ≥ 95% (current ocrmypdf baseline = 100%, but ocrmypdf doesn't produce structured JSON — it produces raw text)
2. Held-out test recall ≥ 90%
3. Inference latency ≤ 5 seconds per PDF on T4 (Cloud Run-friendly)
4. Output JSON validates against the noter schema 99%+ of the time

If criteria 1 or 4 fail, ship as a fallback (when ocrmypdf+regex misses) rather than primary.

## Comparison with the current pipeline

| approach | OCR | schema mapping | structured output | latency | dependency |
|---|---|---|---|---:|---|
| current production | ocrmypdf | nb_sbert + regex | manual assembly | ~2s | local |
| Gemini API | none | semantic | direct | ~10s | external API, $$ |
| **pix2struct_brreg** | **none** | **embedded** | **direct** | **~3s** | **local** |

The pix2struct fine-tune is the path to keeping all noter extraction local while matching Gemini quality.

## Risks

1. **Output format collapse**: model may stop emitting valid JSON if loss optimizer over-rewards token-level recall. Mitigation: validate JSON parseability per epoch, add JSON-validity loss term.
2. **Numeric copy-paste failures**: pix2struct may invent plausible-looking numbers rather than reading the page. Mitigation: cross-validate against tesseract_tsv on the same page; flag any number not present in either.
3. **Norwegian-specific token issues**: pix2struct tokenizer may struggle with æøå. Mitigation: monitor decoded outputs in val set.
