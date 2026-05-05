"""Minimal Colab rig — explicit GCS auth via gcloud."""
import json

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"source":s.splitlines(keepends=True),"execution_count":None,"outputs":[]})

md("""# Minimal Colab rig — iterate on ocr-cascade-eval

**Workflow**: edit code locally → `git push` → re-run cell 1 (clones latest + auth) → re-run cell 2 (your test).

GCS auth uses `gcloud auth application-default login` flow with `--no-browser` for Colab.
Output goes to: `gs://sondre_brreg_data/raw/colab_rig/{session_id}/{ts}_{tag}.{log,json}`
""")

md("## 1 · Setup — auth + clone repo + helper")
code("""# === SETUP — single cell, idempotent ===
PROJECT = 'sondreskarsten-d7d14'
BUCKET  = 'sondre_brreg_data'
RIG_PREFIX = 'raw/colab_rig'

# === Auth (Colab) — explicit GCS scope via google.auth ===
# auth.authenticate_user() alone gives Drive scope, not always GCS — we explicitly
# fetch credentials and pass to storage.Client.
import os, sys, subprocess

try:
    from google.colab import auth as colab_auth
    colab_auth.authenticate_user()  # OAuth popup
    # In Colab, this sets up Application Default Credentials with cloud-platform scope
except ImportError:
    pass  # local Jupyter

subprocess.run(['pip','install','-q','google-cloud-storage','google-auth'], check=False)

# Verify auth + project access by explicitly fetching credentials
import google.auth
try:
    creds, detected_project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/devstorage.read_write'])
    print(f'auth ok — detected project: {detected_project}')
except Exception as e:
    print(f'AUTH FAILED: {type(e).__name__}: {e}')
    print('Fix: Runtime > Disconnect and delete runtime, then re-run this cell.')
    raise

from google.cloud import storage
cli = storage.Client(project=PROJECT, credentials=creds)

# Smoke test: list one blob to verify GCS access
try:
    next(iter(cli.list_blobs(BUCKET, max_results=1)), None)
    print(f'GCS access ok — can list gs://{BUCKET}/')
except Exception as e:
    print(f'GCS access FAILED: {type(e).__name__}: {e}')
    print('Likely cause: your Google account does not have IAM access to project',
          PROJECT, '— check the Colab account in the top-right.')
    raise

# === Clone repo (fresh) ===
subprocess.run(['rm','-rf','/content/ocr-cascade-eval'], check=False)
subprocess.run(['git','clone','-q','https://github.com/sondreskarsten/ocr-cascade-eval.git',
                '/content/ocr-cascade-eval'], check=True)
if '/content/ocr-cascade-eval' not in sys.path:
    sys.path.insert(0, '/content/ocr-cascade-eval')

import time, io, json, traceback, contextlib
SESSION_ID = time.strftime('%Y-%m-%dT%H-%M-%S')
print('\\nrepo head:'); subprocess.run(['git','-C','/content/ocr-cascade-eval','log','--oneline','-1'])
print('session:', SESSION_ID)


def run_and_log(tag, fn, *args, **kwargs):
    \"\"\"Run fn(*args, **kwargs), capture stdout/stderr/result/traceback, ship to GCS.

    Errors do NOT kill the kernel — they're caught and written to the JSON record.
    \"\"\"
    ts = time.strftime('%H-%M-%S')
    log_buf = io.StringIO()
    payload = {'tag': tag, 'session': SESSION_ID, 'started_at': time.time(),
               'args': str(args)[:500], 'kwargs': str(kwargs)[:500]}
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
            result = fn(*args, **kwargs)
        payload['ok'] = True
        try: payload['result'] = json.loads(json.dumps(result, default=str))
        except: payload['result_repr'] = repr(result)[:5000]
    except Exception as e:
        payload['ok'] = False
        payload['error_type'] = type(e).__name__
        payload['error_msg'] = str(e)[:1000]
        payload['traceback'] = traceback.format_exc()[-3000:]
    payload['wall_s'] = round(time.time() - t0, 2)
    payload['stdout_tail'] = log_buf.getvalue()[-8000:]

    log_path = f'{RIG_PREFIX}/{SESSION_ID}/{ts}_{tag}.log'
    json_path = f'{RIG_PREFIX}/{SESSION_ID}/{ts}_{tag}.json'
    cli.bucket(BUCKET).blob(log_path).upload_from_string(payload['stdout_tail'])
    cli.bucket(BUCKET).blob(json_path).upload_from_string(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        content_type='application/json')

    print(f'tag={tag} ok={payload[\"ok\"]} wall={payload[\"wall_s\"]}s')
    print(f'  log:  gs://{BUCKET}/{log_path}')
    print(f'  json: gs://{BUCKET}/{json_path}')
    if not payload['ok']:
        print(f'  ERROR: {payload[\"error_type\"]}: {payload[\"error_msg\"]}')
        print(payload['traceback'][-1500:])
    return payload

print('\\nrig ready — edit cell 2 to run your test')""")

md("""## 2 · RUN — edit this cell to iterate

Pick a pattern. Re-run as often as you like. Errors are caught and written to GCS.""")

code("""# === Pattern A: smoke test ===
def smoke():
    print('hello from colab rig')
    try: import torch; cuda = torch.cuda.is_available()
    except: cuda = None
    return {'cuda': cuda}

run_and_log('smoke', smoke)""")

code("""# === Pattern B: import any module from the repo, call any function ===
import importlib

# Example: render 100-page synthetic corpus from path A
from finetune.tesseract import build_synthetic_corpus
importlib.reload(build_synthetic_corpus)

def build_corpus(n=100, out='/content/corpus_test'):
    blobs = (b for b in cli.list_blobs(BUCKET, prefix='raw/noter_extraction_2025/raw/')
             if b.name.endswith('.json'))
    return build_synthetic_corpus.build_corpus_from_gemini(blobs, out, max_pages=n)

run_and_log('build_corpus_100', build_corpus, n=100)""")

code("""# === Pattern C: run any python file as a subprocess ===
import subprocess
def run_script(args, timeout=1800):
    # Pass GOOGLE_CLOUD_PROJECT so subprocess inherits project context
    env = {**os.environ, 'GOOGLE_CLOUD_PROJECT': PROJECT}
    r = subprocess.run(args, cwd='/content/ocr-cascade-eval',
                       capture_output=True, text=True, timeout=timeout, env=env)
    print('STDOUT:'); print(r.stdout[-3000:])
    print('STDERR:'); print(r.stderr[-3000:])
    return {'returncode': r.returncode}

run_and_log('cli_corpus_100', run_script,
            ['python3', 'finetune/tesseract/build_synthetic_corpus.py',
             '--stage', 'build_corpus',
             '--out_dir', '/content/corpus_cli',
             '--max_pages', '100'])""")

md("## 3 · List artifacts written this session")
code("""for blob in cli.list_blobs(BUCKET, prefix=f'{RIG_PREFIX}/{SESSION_ID}/'):
    print(f'  gs://{BUCKET}/{blob.name}  ({blob.size}b)')""")

nb = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.10'},
        'colab': {'provenance':[]}},
      'nbformat': 4, 'nbformat_minor': 5}
with open('/home/claude/ocr-cascade-eval/notebooks/colab_rig.ipynb','w') as f:
    json.dump(nb, f, indent=1)
print(f'wrote: {len(cells)} cells')
