# Fine-tuning paths

Two fine-tuning paths derived from the v2 cascade audit. They address different parts of the residual gap:

| path | target | input | output | what it fixes |
|---|---|---|---|---|
| **A** | tesseract custom traineddata | synthetic årsregnskap pages | `.traineddata` file | residual TSV miss rate (~1% on v2) |
| **B** | pix2struct fine-tune | 12,879 Gemini extractions | model checkpoint | structured JSON output without OCR pipeline |

## When to start each

| condition | path |
|---|---|
| Cascade is 99/100 reliable but residual misses are critical for production | A |
| Want to replace Gemini API for noter extraction | B |
| Want to ship something fast (~2 days) | A |
| Have a week for higher-leverage work | B |
| Have GPU access (Colab Pro+ A100) | both |
| Have only CPU | A only |

## Sequencing

Path A is faster but lower-leverage. Path B is the right place to invest if the goal is production-replacement of Gemini for noter extraction. Recommended sequence:

1. **Week 1**: Build path A synthetic corpus + start training on Colab. Build path B JSONL in parallel.
2. **Week 2**: Validate path A on v2; if it works, ship `nor_brreg.traineddata`. Continue path B training.
3. **Week 3**: Validate path B on v2 + held-out. If it matches Gemini quality, ship as noter pipeline replacement.

## Both paths share

- v2 fixture as the continuous benchmark (`gs://sondre_brreg_data/raw/ocr_eval_v2_10pdfs_300dpi/`)
- BRREG live API as truth (`/regnskapsregisteret/regnskap/{orgnr}`)
- Training data drawn from the same 12,879 Gemini extractions in `gs://sondre_brreg_data/raw/noter_extraction_2025/raw/`
- Cloud Run for inference deployment after training completes
