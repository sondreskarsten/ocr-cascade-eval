"""Path B — Fine-tune Pix2Struct or Donut on 12,879 Gemini extractions.

Goal: train a single vision-LM that outputs structured JSON directly from page
rasters, replacing the OCR → schema mapping → fill-mask pipeline at inference time.

Why pix2struct over donut:
- pix2struct-docvqa-base already hits 90% on driftsresultat / 70% on årsresultat
  WITHOUT fine-tuning (per llm_audit.md). Donut requires tokenization changes
  for Norwegian.
- pix2struct uses a screenshot-style ViT encoder which matches our PDF page
  raster input. Donut expects document-style images, more sensitive to layout.
- pix2struct's text decoder is general-purpose; we fine-tune it to output JSON.

Training data:
  Input: page raster at 300 DPI
  Target: structured JSON like:
    {"company_name": "BOLIGRADGIVEREN AS",
     "year": 2022,
     "key_metrics": {"sum_eiendeler": 16164967, "aarsresultat": -1101413, ...},
     "noter": [{"note_number": "1", "title": "Lønnskostnader", "full_text": "..."}]}

Source of supervision:
  - Page raster: render from gs://brreg-regnskap/regnskap/{orgnr}/aarsregnskap_{year}.pdf
  - JSON target: gs://sondre_brreg_data/raw/noter_extraction_2025/raw/{orgnr}_aarsregnskap_{year}_v{N}.json
  - Truth verification: BRREG live API key_metrics

Training stages:
  Stage 1: Page-level — train on (single page raster) -> (full JSON for that PDF)
           This trains the model to find any nøkkeltall / note in the visible page.
  Stage 2: Multi-page concatenation — train on (concatenated pages) -> (JSON)
           This teaches the model to handle multi-page documents.
  Stage 3: Schema-strict prompting — add a prompt prefix that requests structured
           output, allowing zero-shot on new firms.

Train/test split:
  - 11,000 PDFs train
  - 1,000 PDFs validation (random sample, stratified by year)
  - 879 PDFs held-out test (touched only after final model is locked)

Hyperparameters:
  - lr 5e-5, batch=4 (A100 80GB), 3 epochs
  - max_input_resolution 4096x4096 (covers A4 at 300 DPI without resize)
  - target_max_tokens 8192
  - mixed precision bf16

Cost estimate: ~1 week wall on A100 80GB Colab Pro+.
- Data prep: 1 day (12,879 PDFs × ~5 pages each = 60K page rasters at 300 DPI ≈ 200 GB)
- Training: 4-5 days (3 epochs over 11K PDFs at ~1 sample/sec on A100)
- Eval: 1 day on test set + held-out
"""
import os
import json
import re
from pathlib import Path


# === Stage 1: Build training set ===

def render_pdf_pages(pdf_blob, dpi=300, max_pages=10):
    """Render PDF pages from GCS blob to PIL images."""
    import fitz
    from PIL import Image
    import io
    
    pdf_bytes = pdf_blob.download_as_bytes()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(min(len(doc), max_pages)):
        page = doc[i]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pages.append(img)
    doc.close()
    return pages


def build_training_jsonl(out_path, max_pdfs=None, sample_seed=42):
    """Iterate over Gemini JSONs, pair with PDFs, write training samples to JSONL.
    
    Each line: {"pdf_blob": "...", "page_n": 3, "target_json": {...}}
    Page-level: each page becomes its own training sample with the full PDF's JSON
    as target (model learns to extract whatever's visible).
    """
    from google.cloud import storage
    cli = storage.Client()
    
    gemini_blobs = list(cli.list_blobs("sondre_brreg_data",
        prefix="raw/noter_extraction_2025/raw/"))
    gemini_blobs = [b for b in gemini_blobs if b.name.endswith(".json")]
    
    import random
    rng = random.Random(sample_seed)
    rng.shuffle(gemini_blobs)
    if max_pdfs: gemini_blobs = gemini_blobs[:max_pdfs]
    
    n_samples = 0
    with open(out_path, "w") as f:
        for blob in gemini_blobs:
            try:
                d = json.loads(blob.download_as_bytes())
            except: continue
            
            orgnr = d.get("orgnr") or d["_orgnr_from_path"]
            year = d.get("year")
            if not orgnr or not year: continue
            
            pdf_blob_name = f"regnskap/{orgnr}/aarsregnskap_{year}.pdf"
            
            # Trim target to schema-stable fields
            target = {
                "orgnr": orgnr,
                "year": year,
                "company_name": d.get("company_name"),
                "ansatte_count": d.get("ansatte_count"),
                "going_concern_mentioned": d.get("going_concern_mentioned"),
                "notes": [{"note_number": n.get("note_number"),
                           "title": n.get("title"),
                           "full_text": n.get("full_text", "")[:2000]}
                          for n in (d.get("notes") or [])],
            }
            
            # Pair each page with the same target (page-level sampling)
            for page_n in range(1, 12):  # we don't know n_pages here; just write 10
                f.write(json.dumps({
                    "pdf_bucket": "brreg-regnskap",
                    "pdf_blob": pdf_blob_name,
                    "page_n": page_n,
                    "target": target,
                }, ensure_ascii=False) + "\n")
                n_samples += 1
            
            if n_samples % 1000 == 0:
                print(f"  wrote {n_samples} samples ...")
    
    print(f"\nTraining JSONL: {out_path} ({n_samples} samples)")
    return n_samples


# === Stage 2: Pix2Struct fine-tuning ===

PIX2STRUCT_TRAINING_SCRIPT = '''
"""Run on Colab Pro+ A100 80GB.

Setup:
  !pip install -q transformers datasets accelerate pillow pymupdf bitsandbytes
  !pip install -q -U google-cloud-storage
  
Train:
  python pix2struct_finetune.py --train_jsonl /content/train.jsonl --epochs 3
"""
import os, json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Pix2StructProcessor, Pix2StructForConditionalGeneration,
    AdamW, get_linear_schedule_with_warmup,
)
from PIL import Image
import io


CKPT = "google/pix2struct-base"  # or "google/pix2struct-docvqa-base" for warm-start


class BRREGPageDataset(Dataset):
    def __init__(self, jsonl_path, processor, gcs_client, max_target_tokens=2048):
        self.lines = open(jsonl_path).readlines()
        self.processor = processor
        self.cli = gcs_client
        self.max_target_tokens = max_target_tokens
        self._pdf_cache = {}

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx):
        sample = json.loads(self.lines[idx])
        pdf_key = (sample["pdf_bucket"], sample["pdf_blob"])
        if pdf_key not in self._pdf_cache:
            pdf_bytes = self.cli.bucket(sample["pdf_bucket"]).blob(
                sample["pdf_blob"]).download_as_bytes()
            self._pdf_cache[pdf_key] = pdf_bytes
            if len(self._pdf_cache) > 5: self._pdf_cache.pop(next(iter(self._pdf_cache)))
        
        import fitz
        doc = fitz.open(stream=self._pdf_cache[pdf_key], filetype="pdf")
        page_idx = sample["page_n"] - 1
        if page_idx >= len(doc):
            doc.close()
            return self.__getitem__((idx + 1) % len(self))
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        doc.close()
        
        target_str = json.dumps(sample["target"], ensure_ascii=False)
        
        inputs = self.processor(
            images=img, text="Extract structured financial data:",
            return_tensors="pt", max_patches=2048,
        )
        labels = self.processor.tokenizer(
            target_str, return_tensors="pt",
            max_length=self.max_target_tokens, padding="max_length",
            truncation=True,
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "flattened_patches": inputs["flattened_patches"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def train(train_jsonl, val_jsonl, output_dir, epochs=3, batch_size=4,
          lr=5e-5, warmup_steps=500):
    from google.cloud import storage
    cli = storage.Client()
    
    processor = Pix2StructProcessor.from_pretrained(CKPT)
    model = Pix2StructForConditionalGeneration.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16)
    model.cuda()
    
    train_ds = BRREGPageDataset(train_jsonl, processor, cli)
    val_ds = BRREGPageDataset(val_jsonl, processor, cli)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        print(f"\\nepoch {epoch}: train_loss={total_loss/len(train_dl):.4f} val_loss={val_loss/len(val_dl):.4f}")
        
        # Checkpoint
        ckpt_dir = f"{output_dir}/epoch_{epoch}"
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"  saved {ckpt_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--output_dir", default="/content/pix2struct_brreg")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()
    train(args.train_jsonl, args.val_jsonl, args.output_dir,
          args.epochs, args.batch_size)
'''


# === Stage 3: Eval against v2 fixture + BRREG truth ===

def eval_pix2struct_brreg(model_dir, fixture_dir):
    """Run fine-tuned model on v2 fixture, parse JSON output, compare to BRREG truth."""
    import torch
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    from PIL import Image
    from google.cloud import storage
    
    processor = Pix2StructProcessor.from_pretrained(model_dir)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_dir).cuda()
    model.eval()
    
    cli = storage.Client()
    truth = {}
    for blob in cli.list_blobs("sondre_brreg_data",
        prefix="raw/ocr_eval_v2_10pdfs_300dpi/audit/brreg_ground_truth/"):
        if blob.name.endswith(".json"):
            d = json.loads(blob.download_as_bytes())
            truth[d["orgnr"]] = d.get("key_metrics", {})
    
    results = {}
    for orgnr, truth_km in truth.items():
        page_imgs = sorted(Path(fixture_dir).glob(f"{orgnr}_p-*.png"))
        # Run inference on each page, merge results
        all_extracted = {}
        for img_path in page_imgs:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, text="Extract structured financial data:",
                               return_tensors="pt", max_patches=2048)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=2048)
            txt = processor.decode(out[0], skip_special_tokens=True)
            try:
                extracted = json.loads(txt)
                if "key_metrics" in extracted:
                    all_extracted.update(extracted["key_metrics"])
            except json.JSONDecodeError:
                pass
        
        # Score against BRREG truth
        hits = 0
        n_truth = 0
        for k, v in truth_km.items():
            if isinstance(v, (int, float)) and v != 0:
                n_truth += 1
                if k in all_extracted and all_extracted[k] == v:
                    hits += 1
        results[orgnr] = {"hits": hits, "n_truth": n_truth,
                           "recall": hits / n_truth if n_truth else None,
                           "extracted": all_extracted}
    
    return {
        "model": "pix2struct_brreg",
        "v2_recall_overall": sum(r["hits"] for r in results.values()) /
                              max(1, sum(r["n_truth"] for r in results.values())),
        "per_pdf": results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build_jsonl", "train", "eval"], required=True)
    parser.add_argument("--out_path", default="/tmp/brreg_train.jsonl")
    parser.add_argument("--max_pdfs", type=int, default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--fixture", default=None)
    args = parser.parse_args()
    
    if args.stage == "build_jsonl":
        build_training_jsonl(args.out_path, max_pdfs=args.max_pdfs)
    elif args.stage == "train":
        print("Run on Colab Pro+ A100. Training script:")
        print(PIX2STRUCT_TRAINING_SCRIPT)
    elif args.stage == "eval":
        if not args.model_dir or not args.fixture:
            raise SystemExit("--model_dir and --fixture required")
        result = eval_pix2struct_brreg(args.model_dir, args.fixture)
        print(json.dumps(result, indent=2, ensure_ascii=False))
