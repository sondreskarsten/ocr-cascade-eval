# Path A: Tesseract custom traineddata

Drop the residual TSV miss rate by training tesseract's LSTM on synthetic Norwegian årsregnskap pages.

## Goal

Tesseract's stock `nor.traineddata` is general-purpose. The misses observed in the v2 cascade audit cluster around:
- Digit cluster recognition (`16 164 967` getting split or merged wrong)
- Parenthesis-negation (`(123)` should be -123)
- Norwegian-specific number formats (space as thousands separator, comma as decimal)

A custom `nor_brreg.traineddata` fine-tuned on synthetic årsregnskap typography should fix these.

## Pipeline

```bash
# Stage 1: Build synthetic corpus (run on Cloud Run or local Python)
python build_synthetic_corpus.py --stage build_corpus \
    --out_dir gs://sondre_brreg_data/raw/tesseract_finetune/synthetic_corpus \
    --max_pages 5000

# Stage 2: Train (run on GPU VM with tesseract-ocr-dev installed)
# - clones tesstrain at runtime
# - uses nor.traineddata as starting point
# - 50,000 iterations LSTM fine-tune
bash lstmtrain.sh \
    /data/synthetic_corpus \
    /data/finetune_out

# Stage 3: Eval on v2 fixture
python build_synthetic_corpus.py --stage eval \
    --traineddata /data/finetune_out/nor_brreg.traineddata \
    --fixture /data/v2_fixture/pages
```

## Expected outcomes

Before fine-tune (current tesseract baseline):
- text-mode: 94/100 = 94% recall
- TSV: 99/100 = 99% recall

After fine-tune (target):
- text-mode: 99-100% recall
- TSV: 99-100% recall (residual ~1 miss should disappear)

If the fine-tune doesn't move the number meaningfully, that's also a useful empirical result — it would tell us the residual misses are caused by image quality / scan artifacts rather than tesseract's vocabulary.

## Cost

- Synthetic corpus generation: ~30 min CPU on 5,000 pages
- LSTM training: 4-12 hours GPU (T4 sufficient), 24-72 hours 8-core CPU
- Eval: ~15 min CPU on v2 fixture

Recommended: use Colab Pro+ A100 for the training step (free with subscription you already have).

## Decision criteria (when to ship)

Ship `nor_brreg.traineddata` to production cascade only if:
1. v2 recall improves AND
2. No regression on a held-out test set of 100 random PDFs from the 12,879
3. Inference time per page stays within 2× of stock tesseract

If the fine-tune fails any of those, keep stock `nor.traineddata` + the existing tesseract_tsv post-processing.
