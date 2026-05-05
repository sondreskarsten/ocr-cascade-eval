"""Build minimal Colab rig notebook.

3 cells:
1. GCS auth (Colab built-in)
2. Clone repo + import any module
3. Run any script and stream stdout/stderr to GCS
"""
import json

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"source":s.splitlines(keepends=True),"execution_count":None,"outputs":[]})

md("""# Minimal Colab rig — iterate on ocr-cascade-eval

Three cells. Run all once, then edit `RUN` cell to iterate.

- **Auth**: Colab built-in Google login (one popup)
- **Repo**: clones `ocr-cascade-eval` fresh on every restart (always pulls latest main)
- **Run**: imports any module, runs any callable, streams output to GCS

Output goes to: `gs://sondre_brreg_data/raw/colab_rig/{session_id}/{timestamp}_{tag}.{log,json}`
""")

md("## 1 · Auth + bucket")
code("""from google.colab import auth as colab_auth
colab_auth.authenticate_user()
!pip install -q google-cloud-storage 2>&1 | tail -1
from google.cloud import storage
PROJECT = 'sondreskarsten-d7d14'
BUCKET = 'sondre_brreg_data'
RIG_PREFIX = 'raw/colab_rig'
cli = storage.Client(project=PROJECT)
print('bucket exists:', cli.bucket(BUCKET).exists())""")

md("""## 2 · Clone repo + helper

Re-run this cell to pull latest main.""")
code("""!cd /content && rm -rf ocr-cascade-eval && git clone -q https://github.com/sondreskarsten/ocr-cascade-eval.git
import sys, os, io, time, json, traceback, contextlib
if '/content/ocr-cascade-eval' not in sys.path:
    sys.path.insert(0, '/content/ocr-cascade-eval')
os.environ['GCS_BUCKET'] = BUCKET
SESSION_ID = time.strftime('%Y-%m-%dT%H-%M-%S')
print('repo at /content/ocr-cascade-eval, head:')
!cd /content/ocr-cascade-eval && git log --oneline -1
print('session:', SESSION_ID)


def run_and_log(tag, fn, *args, **kwargs):
    \"\"\"Run fn(*args, **kwargs), capture stdout+stderr+result+traceback, ship to GCS.\"\"\"
    ts = time.strftime('%H-%M-%S')
    log_buf = io.StringIO()
    payload = {'tag': tag, 'session': SESSION_ID, 'started_at': time.time(),
               'args': str(args)[:500], 'kwargs': str(kwargs)[:500]}
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
            result = fn(*args, **kwargs)
        payload['ok'] = True
        # Try to make result JSON-serialisable
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
        print('  traceback tail:')
        print(payload['traceback'][-1500:])
    return payload""")

md("""## 3 · RUN — edit this cell to iterate

Three patterns. Pick the one matching what you want to test.""")

code("""# === Pattern A: smoke test — just verify the rig works ===
def smoke_test():
    print('hello from colab rig')
    return {'who': 'pix2struct candidate', 'cuda': __import__('torch').cuda.is_available()}

run_and_log('smoke', smoke_test)""")

code("""# === Pattern B: import any module from the repo, call any function ===
# Edit imports + the function call to whatever you want to test.

import importlib
# from runners import tesseract_tsv  # for example
# importlib.reload(tesseract_tsv)

# Example: build 100-page synthetic corpus from path A
from finetune.tesseract import build_synthetic_corpus
importlib.reload(build_synthetic_corpus)

def build_corpus(n=100, out='/content/corpus_test'):
    blobs = (b for b in cli.list_blobs(BUCKET, prefix='raw/noter_extraction_2025/raw/')
             if b.name.endswith('.json'))
    return build_synthetic_corpus.build_corpus_from_gemini(blobs, out, max_pages=n)

run_and_log('build_corpus_100', build_corpus, n=100)""")

code("""# === Pattern C: run any python file as a script via subprocess ===
import subprocess
def run_script(args):
    r = subprocess.run(args, cwd='/content/ocr-cascade-eval',
                       capture_output=True, text=True, timeout=1800)
    print('STDOUT:'); print(r.stdout[-3000:])
    print('STDERR:'); print(r.stderr[-3000:])
    return {'returncode': r.returncode, 'cmd': args}

# example: build 100-page corpus via the actual CLI
run_and_log('cli_corpus_100',
            run_script,
            ['python3', 'finetune/tesseract/build_synthetic_corpus.py',
             '--stage', 'build_corpus',
             '--out_dir', '/content/corpus_cli',
             '--max_pages', '100'])""")

md("""## 4 · Read back from GCS — list all artifacts from this session""")
code("""for blob in cli.list_blobs(BUCKET, prefix=f'{RIG_PREFIX}/{SESSION_ID}/'):
    print(f'  {blob.name}  ({blob.size}b)')""")

md("""## Notes

- **Re-running cell 2** pulls the latest main. So the loop is: edit code locally → push to GitHub → re-run cell 2 → re-run cell 3.
- **Re-running cell 3** with a different tag gives you a separate GCS log entry — your session keeps a full audit trail.
- **`run_and_log`** captures stdout, stderr, traceback, and return value. Errors don't kill the kernel.
- **GPU**: Runtime → Change runtime type → A100 + High-RAM if you're testing pix2struct.
""")

nb = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.10'},
        'colab': {'provenance':[]}},
      'nbformat': 4, 'nbformat_minor': 5}
with open('/home/claude/ocr-cascade-eval/notebooks/colab_rig.ipynb','w') as f:
    json.dump(nb, f, indent=1)
print(f'wrote notebook: {len(cells)} cells')
