"""Build ocr_benchmark_11k.ipynb programmatically — one cell at a time."""
import json

cells = []

def md(s):
    cells.append({"cell_type":"markdown","metadata":{},"source":s.split("\n")})

def code(s):
    cells.append({"cell_type":"code","metadata":{},"source":s.split("\n"),
                  "execution_count":None,"outputs":[]})

# ---------- 1. Header ----------
md("""# OCR / image→text benchmark on 12,879 Norwegian årsregnskap PDFs

**Scope.** Every runner that consumes a page image and produces text or structured-text output. 20 engines:

`tesseract • paddleocr • ocrmypdf • easyocr • doctr • nougat • surya • trocr • olmocr • marker • docling • pix2struct • pixtral • colpali • udop • donut • layoutlmv3 • lilt • camelot • tabula`

**Ground truth.** Gemini 2.5 Flash structured extractions already saved at `gs://sondre_brreg_data/raw/noter_extraction_2025/raw/{orgnr}_aarsregnskap_{year}_v{N}.json` (12,879 files).

**Three OCR-only metrics per (engine × pdf):**
1. **Numeric recall** — fraction of distinct integers from Gemini JSON that appear in the engine's text (whitespace-tolerant: '241101', '241 101', '241.101' all match).
2. **Label recall** — fraction of distinct Gemini line-item labels (length ≥ 4 chars) that appear as substrings in the engine's text.
3. **Distress-phrase recall** — for filings where Gemini set `going_concern_mentioned=True`, fraction of standard Norwegian distress phrases recovered.

**Restartable.** Per-engine state at `gs://sondre_brreg_data/raw/ocr_bench_11k/{engine}/{orgnr}.json`. Re-running picks up from the next unprocessed orgnr automatically.""")

# ---------- 2. Auth ----------
md("""## 1 · Auth + GCS setup

Upload the service account key file `sondreskarsten-d7d14-8486be2d085b.json`. In Colab use the file-upload widget; in a local Jupyter put the key beside the notebook.""")

code("""# Auth: file upload (Colab) or pre-existing path (local Jupyter)
import os, sys, pathlib

KEY_DEFAULT = "/content/sondreskarsten-d7d14-8486be2d085b.json"
if not os.path.exists(KEY_DEFAULT):
    try:
        from google.colab import files  # noqa
        print("Upload sondreskarsten-d7d14-8486be2d085b.json …")
        files.upload()
    except ImportError:
        cwd_key = pathlib.Path.cwd() / "sondreskarsten-d7d14-8486be2d085b.json"
        if cwd_key.exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(cwd_key)
        else:
            raise SystemExit("Place sondreskarsten-d7d14-8486be2d085b.json next to this notebook")

if os.path.exists(KEY_DEFAULT):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_DEFAULT

print("GOOGLE_APPLICATION_CREDENTIALS=", os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))

# Sanity-check GCS access
!pip install -q google-cloud-storage google-auth 2>&1 | tail -1
from google.cloud import storage
cli = storage.Client()
print("project:", cli.project)
print("bucket sondre_brreg_data exists:", cli.bucket('sondre_brreg_data').exists())""")

# ---------- 3. Clone + GPU check ----------
md("""## 2 · Clone the eval repo + GPU check""")

code("""!cd /content && rm -rf ocr-cascade-eval
!cd /content && git clone -q https://github.com/sondreskarsten/ocr-cascade-eval.git
import sys
sys.path.insert(0, '/content/ocr-cascade-eval')
print("repo at /content/ocr-cascade-eval, head:")
!cd /content/ocr-cascade-eval && git log --oneline -1

# GPU check
import torch
print()
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("VRAM total MiB:", torch.cuda.get_device_properties(0).total_memory // (1024*1024))""")

# ---------- 4. Manifest ----------
md("""## 3 · Build manifest of 11k PDFs to benchmark

Manifest = every `(orgnr, year)` pair where both the PDF and a Gemini JSON exist. Cached after first build.""")

code("""import json, os
from google.cloud import storage

cli = storage.Client()
MANIFEST_LOCAL = '/content/ocr_bench_manifest.json'

def build_manifest():
    out = []
    # Iterate Gemini JSONs
    for blob in cli.list_blobs('sondre_brreg_data', prefix='raw/noter_extraction_2025/raw/'):
        n = blob.name.split('/')[-1]
        if not n.endswith('.json'):
            continue
        # Pattern: 811573272_aarsregnskap_2024_v2.json
        parts = n.replace('.json','').split('_')
        if len(parts) < 4 or parts[1] != 'aarsregnskap':
            continue
        orgnr, year = parts[0], parts[2]
        if not (orgnr.isdigit() and year.isdigit()): continue
        out.append({"orgnr": orgnr, "year": int(year),
                    "gemini_json": f"gs://sondre_brreg_data/{blob.name}",
                    "pdf": f"gs://brreg-regnskap/regnskap/{orgnr}/aarsregnskap_{year}.pdf"})
    return out

if os.path.exists(MANIFEST_LOCAL):
    manifest = json.load(open(MANIFEST_LOCAL))
    print(f"loaded cached manifest: {len(manifest)} pdfs")
else:
    manifest = build_manifest()
    json.dump(manifest, open(MANIFEST_LOCAL,'w'))
    print(f"built manifest: {len(manifest)} pdfs")

# Stable ordering for resumability
manifest.sort(key=lambda r: (r['orgnr'], r['year']))
print(f"first 3: {manifest[:3]}")""")

# ---------- 5. Scorer ----------
md("""## 4 · OCR scoring function

Three recall metrics, all computed against the Gemini JSON for that filing.""")

code("""import re, json

DISTRESS_PHRASES = [
    'fortsatt drift',
    'usikkerhet om fortsatt drift',
    'Egenkapitalen er tapt',
    'negativ egenkapital',
    'gjeldsforhandling',
]

NUM_RE = re.compile(r'-?\\d+')

def number_present(n, text):
    \"\"\"True if integer n appears in text under any whitespace formatting (Norwegian thousands sep).\"\"\"
    if n is None: return False
    n = int(round(n))
    sign = '-' if n < 0 else ''
    n_abs = abs(n)
    s = str(n_abs)
    if n_abs == 0:
        return ' 0' in text or text.startswith('0') or '\\n0' in text
    s_rev = s[::-1]
    groups = [s_rev[i:i+3][::-1] for i in range(0, len(s_rev), 3)][::-1]
    pat_grouped = sign + r'\\s*'.join(re.escape(g) for g in groups)
    pat_digits = sign + r'\\s*'.join(re.escape(d) for d in s)
    return bool(re.search(pat_grouped, text)) or bool(re.search(pat_digits, text))


def collect_gemini_signal(gemini_json):
    \"\"\"Pull (numbers, labels, distress_present) from a Gemini extraction.\"\"\"
    nums, labels = set(), set()
    for arr_name in ('resultatregnskap','balanse_eiendeler',
                     'balanse_egenkapital_og_gjeld','kontantstrom'):
        for item in (gemini_json.get(arr_name) or []):
            for k in ('amount_year','amount_prior_year'):
                v = item.get(k)
                if isinstance(v,(int,float)) and v != 0:
                    nums.add(int(round(v)))
            lab = (item.get('label') or '').strip()
            if len(lab) >= 4:
                labels.add(lab)
    for note in (gemini_json.get('noter') or []):
        title = (note.get('title') or '').strip()
        if len(title) >= 4: labels.add(title)
    distress_present = bool(gemini_json.get('going_concern_mentioned'))
    return nums, labels, distress_present


def score_ocr_text(ocr_text, gemini_json):
    nums, labels, distress_present = collect_gemini_signal(gemini_json)
    n_num_total = len(nums)
    n_num_hit = sum(1 for n in nums if number_present(n, ocr_text))
    n_lab_total = len(labels)
    # case-insensitive substring
    text_lower = ocr_text.lower()
    n_lab_hit = sum(1 for l in labels if l.lower() in text_lower)
    # distress phrases (only meaningful if Gemini flagged distress)
    if distress_present:
        n_distress_hit = sum(1 for ph in DISTRESS_PHRASES
                             if ph.lower() in text_lower)
        distress_recall = n_distress_hit / len(DISTRESS_PHRASES)
    else:
        distress_recall = None
    # Extra numbers in OCR not in Gemini (size of "noise floor")
    ocr_nums = set()
    for tok in NUM_RE.findall(re.sub(r'(?<=\\d)\\s+(?=\\d)', '', ocr_text)):  # collapse digit groups
        try:
            v = int(tok)
            if abs(v) >= 1000:  # skip page numbers / small noise
                ocr_nums.add(v)
        except: pass
    extra_in_ocr = len(ocr_nums - nums)
    return {
        "n_num_total": n_num_total,
        "n_num_hit": n_num_hit,
        "numeric_recall": n_num_hit / n_num_total if n_num_total else None,
        "n_lab_total": n_lab_total,
        "n_lab_hit": n_lab_hit,
        "label_recall": n_lab_hit / n_lab_total if n_lab_total else None,
        "n_chars_ocr": len(ocr_text),
        "n_extra_numbers_in_ocr": extra_in_ocr,
        "distress_present_in_gemini": distress_present,
        "distress_recall": distress_recall,
    }


# Self-test against fixture v2
print("self-test on FARBOSS:")
import urllib.request
url = "https://storage.googleapis.com/sondre_brreg_data/raw/ocr_eval_v2_10pdfs_300dpi/audit/gemini_ground_truth/814747352.json"
gem = json.loads(cli.bucket('sondre_brreg_data').blob('raw/ocr_eval_v2_10pdfs_300dpi/audit/gemini_ground_truth/814747352.json').download_as_bytes())
fake_ocr = "FARBOSS AS årsresultat 241 101 sum eiendeler 681 828 driftsresultat -8 900 sum kostnader 8 900"
sc = score_ocr_text(fake_ocr, gem)
print(json.dumps(sc, indent=2))""")

# ---------- 6. Per-engine runner ----------
md("""## 5 · Per-engine runner with skip-already-done logic

Each engine has its own runner module in `runners/`. We import it, install its deps once, then iterate over the manifest. Results upload to `gs://sondre_brreg_data/raw/ocr_bench_11k/{engine}/{orgnr}.json`.""")

code("""import json, os, time, traceback, gc
from pathlib import Path
from google.cloud import storage
import fitz  # PyMuPDF

cli = storage.Client()
RESULTS_PREFIX = 'raw/ocr_bench_11k'  # bench namespace
DPI = 300
LOCAL_TMP = Path('/content/tmp_ocr_bench')
LOCAL_TMP.mkdir(parents=True, exist_ok=True)


def already_done(engine, orgnr):
    return cli.bucket('sondre_brreg_data').blob(
        f"{RESULTS_PREFIX}/{engine}/{orgnr}.json").exists()


def upload_result(engine, orgnr, payload):
    cli.bucket('sondre_brreg_data').blob(
        f"{RESULTS_PREFIX}/{engine}/{orgnr}.json").upload_from_string(
            json.dumps(payload, ensure_ascii=False, default=str),
            content_type='application/json')


def render_pdf_to_pngs(gs_pdf_path, out_dir, dpi=DPI):
    \"\"\"Download PDF → render every page at dpi to PNG.\"\"\"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bucket = gs_pdf_path.split('/')[2]
    object_path = '/'.join(gs_pdf_path.split('/')[3:])
    pdf_local = out_dir / 'src.pdf'
    cli.bucket(bucket).blob(object_path).download_to_filename(str(pdf_local))
    doc = fitz.open(str(pdf_local))
    page_paths, page_size = [], []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        p = out_dir / f'p-{i+1:02d}.png'
        pix.save(str(p))
        page_paths.append(str(p))
        page_size.append([pix.width, pix.height])
    doc.close()
    return str(pdf_local), page_paths, page_size


def fetch_gemini(orgnr, year):
    # Gemini JSONs follow the pattern {orgnr}_aarsregnskap_{year}_v{N}.json — find latest version
    blobs = list(cli.list_blobs('sondre_brreg_data',
        prefix=f'raw/noter_extraction_2025/raw/{orgnr}_aarsregnskap_{year}_'))
    if not blobs: return None
    # latest version (highest v number)
    blobs.sort(key=lambda b: int(b.name.split('_v')[-1].replace('.json','')))
    return json.loads(blobs[-1].download_as_bytes())


def benchmark_one(engine_fn, engine_name, manifest_row):
    orgnr = manifest_row['orgnr']
    year = manifest_row['year']
    if already_done(engine_name, orgnr):
        return {"orgnr": orgnr, "skipped": True}
    t0 = time.time()
    work = LOCAL_TMP / f'{engine_name}_{orgnr}'
    try:
        pdf_local, page_paths, page_size = render_pdf_to_pngs(
            manifest_row['pdf'], work, dpi=DPI)
    except Exception as e:
        payload = {"orgnr": orgnr, "year": year, "stage": "render",
                   "error": f"{type(e).__name__}: {str(e)[:200]}",
                   "wall_s": time.time()-t0}
        upload_result(engine_name, orgnr, payload)
        return payload
    # Build a bundle compatible with engine signatures
    bundle = {
        "pdf": pdf_local,
        "page_imgs": page_paths,
        "page_size": page_size,
        "n_pages": len(page_paths),
        "page_text": {},  # OCR engines produce this
        "page_words": {},
        "full_text": "",
        "orgnr": orgnr, "year": year,
    }
    try:
        out = engine_fn(orgnr, bundle)
    except Exception as e:
        payload = {"orgnr": orgnr, "year": year, "stage": "engine",
                   "error": f"{type(e).__name__}: {str(e)[:300]}",
                   "traceback": traceback.format_exc()[-500:],
                   "wall_s": time.time()-t0}
        upload_result(engine_name, orgnr, payload)
        # cleanup
        for p in work.iterdir(): p.unlink(missing_ok=True)
        work.rmdir()
        return payload
    # Get OCR text out of the engine output
    ocr_text = out.get("full_text") or out.get("text") or ""
    if not ocr_text and "pages" in out:
        ocr_text = "\\n".join(p.get("text","") for p in out["pages"])
    # Score
    gem = fetch_gemini(orgnr, year)
    scores = score_ocr_text(ocr_text, gem) if gem else None
    payload = {
        "orgnr": orgnr, "year": year,
        "engine": engine_name,
        "wall_s": round(time.time() - t0, 2),
        "n_chars_ocr_text": len(ocr_text),
        "n_pages": len(page_paths),
        "engine_output_keys": list(out.keys()),
        "scores": scores,
        "full_text": ocr_text[:50000],  # cap stored text at 50k chars
    }
    upload_result(engine_name, orgnr, payload)
    # cleanup
    for p in work.iterdir():
        try: p.unlink()
        except: pass
    try: work.rmdir()
    except: pass
    gc.collect()
    return payload""")

# ---------- 7. Engine adapters ----------
md("""## 6 · Engine adapters (image → text only)

Each adapter installs its deps lazily, returns a callable `(orgnr, bundle) → {full_text, ...}`. Reads the page PNGs in `bundle['page_imgs']`. All 20 image-consuming runners are below; comment out any you don't want.""")

code("""# === lightweight CPU OCR engines ===

def make_tesseract():
    import subprocess
    subprocess.run(['apt-get','-qq','install','-y','tesseract-ocr','tesseract-ocr-nor'],
                   check=False, capture_output=True)
    !pip install -q pytesseract 2>&1 | tail -1
    import pytesseract
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            txt = pytesseract.image_to_string(p, lang='nor')
            pages.append({"text": txt})
        return {"full_text": "\\n".join(pg['text'] for pg in pages),
                "pages": pages, "n_pages": len(pages)}
    return fn


def make_easyocr():
    !pip install -q easyocr 2>&1 | tail -1
    import easyocr
    reader = easyocr.Reader(['no','en'], gpu=torch.cuda.is_available())
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            res = reader.readtext(p, detail=0, paragraph=True)
            pages.append({"text": "\\n".join(res)})
        return {"full_text": "\\n".join(pg['text'] for pg in pages),
                "pages": pages, "n_pages": len(pages)}
    return fn


def make_paddleocr():
    !pip install -q paddleocr paddlepaddle-gpu 2>&1 | tail -1
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang='en')
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            r = ocr.ocr(p, cls=False)
            lines = []
            if r and r[0]:
                for line in r[0]:
                    if len(line) >= 2 and isinstance(line[1], (list,tuple)):
                        lines.append(line[1][0])
            pages.append({"text": "\\n".join(lines)})
        return {"full_text": "\\n".join(pg['text'] for pg in pages),
                "pages": pages, "n_pages": len(pages)}
    return fn


def make_ocrmypdf():
    import subprocess
    subprocess.run(['apt-get','-qq','install','-y','ocrmypdf','tesseract-ocr-nor'],
                   check=False, capture_output=True)
    !pip install -q pdfplumber 2>&1 | tail -1
    import pdfplumber, tempfile
    def fn(orgnr, b):
        out_pdf = b['pdf'].replace('.pdf','_ocr.pdf')
        subprocess.run(['ocrmypdf','--force-ocr','-l','nor',b['pdf'],out_pdf],
                       check=False, capture_output=True)
        if not os.path.exists(out_pdf):
            return {"full_text": "", "pages": [], "n_pages": 0,
                    "error": "ocrmypdf failed"}
        with pdfplumber.open(out_pdf) as pdf:
            pages = [{"text": pg.extract_text() or ""} for pg in pdf.pages]
        return {"full_text": "\\n".join(pg['text'] for pg in pages),
                "pages": pages, "n_pages": len(pages)}
    return fn""")

code("""# === GPU-friendly OCR / vision engines ===

def make_doctr():
    !pip install -q python-doctr[torch] 2>&1 | tail -1
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    model = ocr_predictor(pretrained=True)
    def fn(orgnr, b):
        # doctr can handle PDF directly
        doc = DocumentFile.from_pdf(b['pdf'])
        res = model(doc)
        pages = []
        for page in res.pages:
            lines = []
            for block in page.blocks:
                for line in block.lines:
                    lines.append(' '.join(w.value for w in line.words))
            pages.append({"text": "\\n".join(lines)})
        return {"full_text": "\\n".join(pg['text'] for pg in pages),
                "pages": pages, "n_pages": len(pages)}
    return fn


def make_nougat():
    !pip install -q nougat-ocr 2>&1 | tail -1
    from nougat import NougatModel
    from nougat.utils.dataset import LazyDataset
    from nougat.utils.checkpoint import get_checkpoint
    from nougat.postprocessing import markdown_compatible
    import torch
    ckpt = get_checkpoint('0.1.0-base')
    model = NougatModel.from_pretrained(ckpt)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    def fn(orgnr, b):
        from PIL import Image
        pages = []
        for img_path in b['page_imgs']:
            img = Image.open(img_path).convert('RGB')
            pixel_values = model.encoder.prepare_input(img, random_padding=False).unsqueeze(0)
            if torch.cuda.is_available(): pixel_values = pixel_values.cuda()
            with torch.no_grad():
                out = model.inference(image_tensors=pixel_values)
            md = markdown_compatible(out['predictions'][0])
            pages.append({"text": md})
        return {"full_text": "\\n".join(pg['text'] for pg in pages),
                "pages": pages, "n_pages": len(pages)}
    return fn


def make_trocr():
    !pip install -q transformers 2>&1 | tail -1
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    proc = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    if torch.cuda.is_available(): model = model.cuda()
    def fn(orgnr, b):
        pages = []
        for img_path in b['page_imgs']:
            img = Image.open(img_path).convert('RGB')
            pv = proc(images=img, return_tensors='pt').pixel_values
            if torch.cuda.is_available(): pv = pv.cuda()
            with torch.no_grad():
                ids = model.generate(pv, max_length=512)
            txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
            pages.append({"text": txt})
        return {"full_text": "\\n".join(pg['text'] for pg in pages),
                "pages": pages, "n_pages": len(pages)}
    return fn


def make_surya():
    !pip install -q surya-ocr 2>&1 | tail -1
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det, load_processor as load_det_proc
    from surya.model.recognition.model import load_model as load_rec
    from surya.model.recognition.processor import load_processor as load_rec_proc
    from PIL import Image
    det_m, det_p = load_det(), load_det_proc()
    rec_m, rec_p = load_rec(), load_rec_proc()
    def fn(orgnr, b):
        imgs = [Image.open(p).convert('RGB') for p in b['page_imgs']]
        langs = [['no','en']] * len(imgs)
        preds = run_ocr(imgs, langs, det_m, det_p, rec_m, rec_p)
        pages = []
        for pred in preds:
            lines = [tl.text for tl in pred.text_lines]
            pages.append({"text": "\\n".join(lines)})
        return {"full_text": "\\n".join(pg['text'] for pg in pages),
                "pages": pages, "n_pages": len(pages)}
    return fn


def make_olmocr():
    !pip install -q olmocr 2>&1 | tail -1
    # olmocr is heavy; falls through if unavailable
    try:
        from olmocr.pipeline import build_page_query
        # placeholder: real olmocr requires a vLLM server
        def fn(orgnr, b):
            return {"full_text": "", "note": "olmocr requires vLLM server; not run inline"}
        return fn
    except Exception as e:
        def fn(orgnr, b):
            return {"full_text": "", "error": str(e)[:200]}
        return fn


def make_marker():
    !pip install -q marker-pdf 2>&1 | tail -1
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    converter = PdfConverter(artifact_dict=create_model_dict())
    def fn(orgnr, b):
        rendered = converter(b['pdf'])
        text, _, _ = text_from_rendered(rendered)
        return {"full_text": text}
    return fn


def make_docling():
    !pip install -q docling 2>&1 | tail -1
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    def fn(orgnr, b):
        r = conv.convert(b['pdf'])
        md = r.document.export_to_markdown()
        return {"full_text": md}
    return fn""")

code("""# === Vision-language and document-understanding engines ===

def make_pix2struct():
    !pip install -q transformers 2>&1 | tail -1
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    from PIL import Image
    ckpt = 'google/pix2struct-docvqa-base'
    proc = Pix2StructProcessor.from_pretrained(ckpt)
    model = Pix2StructForConditionalGeneration.from_pretrained(ckpt)
    if torch.cuda.is_available(): model = model.cuda()
    QS = ['What is the company name?','What is the year?','What is the årsresultat?',
          'What is the sum eiendeler?','What is the sum egenkapital?',
          'What is the driftsresultat?','What is the sum kostnader?']
    def fn(orgnr, b):
        # Run questions on key pages (2-4 typically)
        n = len(b['page_imgs'])
        targets = b['page_imgs'][1:min(5,n)] if n >= 2 else b['page_imgs']
        out_lines = []
        for img_path in targets:
            img = Image.open(img_path).convert('RGB')
            for q in QS:
                inputs = proc(images=img, return_tensors='pt', text=q)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k,v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=48)
                a = proc.decode(out[0], skip_special_tokens=True)
                out_lines.append(f"{q} {a}")
        return {"full_text": "\\n".join(out_lines)}
    return fn


def make_donut():
    !pip install -q transformers 2>&1 | tail -1
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image
    proc = DonutProcessor.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')
    model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')
    if torch.cuda.is_available(): model = model.cuda()
    def fn(orgnr, b):
        out_pages = []
        for img_path in b['page_imgs']:
            img = Image.open(img_path).convert('RGB')
            pv = proc(img, return_tensors='pt').pixel_values
            if torch.cuda.is_available(): pv = pv.cuda()
            decoder_ids = proc.tokenizer('<s_cord-v2>', add_special_tokens=False, return_tensors='pt').input_ids
            if torch.cuda.is_available(): decoder_ids = decoder_ids.cuda()
            with torch.no_grad():
                out = model.generate(pv, decoder_input_ids=decoder_ids, max_length=512,
                                     bad_words_ids=[[proc.tokenizer.unk_token_id]],
                                     return_dict_in_generate=True)
            txt = proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
            out_pages.append({"text": txt})
        return {"full_text": "\\n".join(p['text'] for p in out_pages), "pages": out_pages}
    return fn


def make_udop():
    !pip install -q transformers 2>&1 | tail -1
    from transformers import UdopProcessor, UdopForConditionalGeneration
    from PIL import Image
    proc = UdopProcessor.from_pretrained('microsoft/udop-large')
    model = UdopForConditionalGeneration.from_pretrained('microsoft/udop-large')
    if torch.cuda.is_available(): model = model.cuda()
    def fn(orgnr, b):
        # UDOP needs bbox+words; we feed empty word list and rely on visual OCR head
        pages = []
        for img_path in b['page_imgs']:
            img = Image.open(img_path).convert('RGB')
            try:
                inputs = proc(img, ['Question answering. Extract all financial numbers and labels from this page.'],
                              return_tensors='pt', text_pair=[''])
                if torch.cuda.is_available():
                    inputs = {k: (v.cuda() if hasattr(v,'cuda') else v) for k,v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=512)
                txt = proc.tokenizer.decode(out[0], skip_special_tokens=True)
                pages.append({"text": txt})
            except Exception as e:
                pages.append({"text": "", "error": str(e)[:120]})
        return {"full_text": "\\n".join(p.get('text','') for p in pages), "pages": pages}
    return fn


def make_layoutlmv3():
    # LayoutLMv3 has no text-decoder head — use it as feature-only on words from tesseract.
    # Returns the merged tesseract+positional output (essentially: tesseract pass-through with bbox tags).
    !pip install -q transformers pytesseract 2>&1 | tail -1
    import pytesseract
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            t = pytesseract.image_to_string(p, lang='nor')
            pages.append({"text": t})
        return {"full_text": "\\n".join(pg['text'] for pg in pages), "pages": pages,
                "note": "layoutlmv3 has no text-gen head; using tesseract for textual content"}
    return fn


def make_lilt():
    # LiLT is layout-aware classifier; without finetuning it can't OCR. Use tesseract as text extractor.
    !pip install -q transformers pytesseract 2>&1 | tail -1
    import pytesseract
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            t = pytesseract.image_to_string(p, lang='nor')
            pages.append({"text": t})
        return {"full_text": "\\n".join(pg['text'] for pg in pages), "pages": pages,
                "note": "LiLT has no decoder head; using tesseract for textual content"}
    return fn


def make_pixtral():
    # 12B vision LM — typically too large for Colab Pro single GPU. Document and skip.
    def fn(orgnr, b):
        return {"full_text": "",
                "note": "Pixtral-12B requires >24GB VRAM; not run inline in benchmark"}
    return fn


def make_colpali():
    # ColPali produces image embeddings, not text. Skip from OCR benchmark.
    def fn(orgnr, b):
        return {"full_text": "", "note": "ColPali emits embeddings, not text"}
    return fn""")

code("""# === Table extractors with text fall-back ===

def make_camelot():
    !pip install -q camelot-py[cv] 2>&1 | tail -1
    import camelot
    def fn(orgnr, b):
        try:
            tables = camelot.read_pdf(b['pdf'], pages='all', flavor='lattice')
        except Exception:
            try: tables = camelot.read_pdf(b['pdf'], pages='all', flavor='stream')
            except Exception as e:
                return {"full_text": "", "error": str(e)[:120]}
        all_txt = []
        for t in tables:
            for _, row in t.df.iterrows():
                all_txt.extend(str(c) for c in row.tolist())
        return {"full_text": "\\n".join(all_txt), "n_tables": len(tables)}
    return fn


def make_tabula():
    !pip install -q tabula-py 2>&1 | tail -1
    import tabula
    def fn(orgnr, b):
        try:
            tables = tabula.read_pdf(b['pdf'], pages='all', stream=True)
        except Exception as e:
            return {"full_text": "", "error": str(e)[:120]}
        all_txt = []
        for df in tables:
            for _, row in df.iterrows():
                all_txt.extend(str(c) for c in row.tolist())
        return {"full_text": "\\n".join(all_txt), "n_tables": len(tables)}
    return fn""")

# ---------- 8. Driver ----------
md("""## 7 · Driver — pick engines, batch size, run

Modify `ENGINES` and `BATCH_SIZE` to suit your VRAM. The driver iterates `manifest`, skips already-done, processes one engine at a time (loads model once, then sweeps PDFs).""")

code("""# ENGINES to run — comment out anything you don't want
ENGINES = {
    'tesseract':    make_tesseract,
    'easyocr':      make_easyocr,
    'paddleocr':    make_paddleocr,
    'ocrmypdf':     make_ocrmypdf,
    'doctr':        make_doctr,
    'nougat':       make_nougat,
    'trocr':        make_trocr,
    'surya':        make_surya,
    'marker':       make_marker,
    'docling':      make_docling,
    'pix2struct':   make_pix2struct,
    'donut':        make_donut,
    'udop':         make_udop,
    'layoutlmv3':   make_layoutlmv3,
    'lilt':         make_lilt,
    'camelot':      make_camelot,
    'tabula':       make_tabula,
    # large/blocked — disabled by default:
    # 'pixtral':     make_pixtral,
    # 'colpali':     make_colpali,
    # 'olmocr':      make_olmocr,
}

# How many PDFs per engine in this run? Set to None for ALL 12,879
MAX_PDFS = 100

import time
for engine_name, factory in ENGINES.items():
    print(f"\\n=== ENGINE: {engine_name} ===")
    # Skip if all manifest items already done for this engine
    blob_iter = cli.list_blobs('sondre_brreg_data',
        prefix=f'{RESULTS_PREFIX}/{engine_name}/', max_results=20000)
    done = {b.name.split('/')[-1].replace('.json','') for b in blob_iter}
    pending = [r for r in manifest if r['orgnr'] not in done]
    if MAX_PDFS:
        pending = pending[:MAX_PDFS]
    print(f"  done={len(done)} | pending in this run={len(pending)}")
    if not pending:
        continue
    t0 = time.time()
    try:
        engine_fn = factory()
    except Exception as e:
        print(f"  FACTORY FAILED: {type(e).__name__}: {str(e)[:180]}")
        continue
    print(f"  factory init: {time.time()-t0:.1f}s")
    for i, row in enumerate(pending):
        try:
            r = benchmark_one(engine_fn, engine_name, row)
            sc = r.get('scores') or {}
            print(f"  [{i+1:>4}/{len(pending)}] {row['orgnr']} y={row['year']} "
                  f"chars={r.get('n_chars_ocr_text','?')} "
                  f"num={sc.get('numeric_recall')} lab={sc.get('label_recall')} "
                  f"wall={r.get('wall_s','?')}s")
        except Exception as e:
            print(f"  [{i+1:>4}] {row['orgnr']} HARDFAIL: {type(e).__name__}: {str(e)[:120]}")
    # Free GPU memory between engines
    try:
        del engine_fn
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass""")

# ---------- 9. Aggregator ----------
md("""## 8 · Aggregate — per-engine summary table""")

code("""import json
from collections import defaultdict

agg = defaultdict(lambda: {
    'n_pdfs': 0, 'n_with_scores': 0,
    'sum_num_recall': 0.0, 'sum_lab_recall': 0.0,
    'sum_distress_recall': 0.0, 'n_distress_eligible': 0,
    'wall_total_s': 0.0, 'errors': 0,
})

for engine in ENGINES:
    for blob in cli.list_blobs('sondre_brreg_data', prefix=f'{RESULTS_PREFIX}/{engine}/'):
        if not blob.name.endswith('.json'): continue
        try:
            d = json.loads(blob.download_as_bytes())
        except: continue
        a = agg[engine]
        a['n_pdfs'] += 1
        a['wall_total_s'] += d.get('wall_s') or 0
        if d.get('error'):
            a['errors'] += 1
            continue
        sc = d.get('scores') or {}
        if sc.get('numeric_recall') is not None:
            a['n_with_scores'] += 1
            a['sum_num_recall'] += sc['numeric_recall']
            a['sum_lab_recall'] += sc['label_recall'] or 0
            if sc.get('distress_recall') is not None:
                a['n_distress_eligible'] += 1
                a['sum_distress_recall'] += sc['distress_recall']

print(f"{'engine':<14} {'n':<6} {'n_score':<8} {'avg_num':<10} {'avg_lab':<10} {'avg_distr':<10} {'wall_total':<10} {'err':<5}")
print('-'*90)
for engine, a in sorted(agg.items()):
    n = a['n_with_scores']
    avg_num = a['sum_num_recall']/n if n else 0
    avg_lab = a['sum_lab_recall']/n if n else 0
    avg_distr = a['sum_distress_recall']/a['n_distress_eligible'] if a['n_distress_eligible'] else 0
    print(f"{engine:<14} {a['n_pdfs']:<6} {n:<8} {avg_num:<10.3f} {avg_lab:<10.3f} "
          f"{avg_distr:<10.3f} {a['wall_total_s']:<10.0f} {a['errors']:<5}")
""")

# ---------- 10. Save aggregate to GCS ----------
md("""## 9 · Save the aggregate to GCS""")

code("""import json, time
out = {'generated_at': time.time(),
       'manifest_n': len(manifest), 'engines_run': list(ENGINES.keys()),
       'aggregates': {k: dict(v) for k,v in agg.items()}}
cli.bucket('sondre_brreg_data').blob(
    f'{RESULTS_PREFIX}/_aggregate_latest.json').upload_from_string(
        json.dumps(out, ensure_ascii=False, indent=2),
        content_type='application/json')
print(f"saved gs://sondre_brreg_data/{RESULTS_PREFIX}/_aggregate_latest.json")""")

# ---------- assemble notebook ----------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
with open('/home/claude/ocr-cascade-eval/notebooks/ocr_benchmark_11k.ipynb','w') as f:
    json.dump(nb, f, indent=1)
print(f"wrote notebook: {len(json.dumps(nb))} bytes, {len(cells)} cells")
