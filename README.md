# News Topic Classification

A four-topic classifier with a Next.js website and FastAPI inference backend,
prepared for two Vercel projects. The former Streamlit interface is archived in
`legacy/`.

The implementation has **not been trained, tested, built, or deployed**. Historical
DistilBERT test macro F1 is approximately 93.53%; **95%+ remains a target**.
Run the commands below yourself. New results and live URLs do not exist yet.

## Training and test data are separate

- `data/Training_data_9.csv` is the only CSV read by `train`, `tune`, and `benchmark`.
- `data/Test_data.csv` is read only by explicit final evaluation commands. It is
  never used for training, cleaning training data, fitting features, architecture
  selection, early stopping, hyperparameter search, or deployment optimization.
- Training normalizes duplicate identities, removes conflicting training-label
  groups, and persists a shared stratified 80/20 split. Original CSVs are unchanged.
- The historical audited cleaned file is deliberately not reused: its creation
  consulted test identities. The new workflow cleans the original training CSV alone.
- Training does not inspect test overlap. A separate audit would be needed to
  establish whether the supplied files contain overlapping examples.
- This test set has been evaluated historically; describe scores as benchmark
  results, not evidence from a previously untouched holdout.

## Setup (PowerShell, project root)

Use Python 3.12 and Node.js 20.9+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For GPU training, use a CUDA-enabled PyTorch build compatible with your driver:
[official installer](https://pytorch.org/get-started/locally/).
DistilBERT and Google News Word2Vec download pretrained resources on first use.
Allow disk space for their caches and per-trial checkpoints.

## 1. Train and tune ? training CSV only

```powershell
python main.py tune --config configs/improve.yaml --budget-hours 24 --output results/improvement
```

Repeating the command skips completed trials. Interrupted/failed trials restart
from their configuration, not from optimizer state. The study retains consumed
phase time; increase `--budget-hours` to extend it. Use a new output directory if
changing configuration or data.

The budget allocates eight hours to baseline/preprocessing screening, eight to
architecture/hyperparameter candidates, and six to seed confirmation. The
remaining two hours are reserved for your separately invoked evaluation/report
work: tuning never automatically evaluates test data. Six upgraded trials per
family are planned by default; change this with `--trials-per-family`.

Review:

- `results/improvement/study.json`: configurations, statuses, timing, validation results.
- `results/improvement/split_manifest.json`: shared training/validation partition.
- `results/improvement/selection.json`: per-family selections and the overall
  validation-selected winner. Check `missing_families` and `confirmation_complete`.
- Per-trial folders: logs, configuration, selected-epoch metrics, history and weights.

To train just one upgraded model:

```powershell
python main.py train --config configs/improve.yaml --model distilbert --preprocessing raw
```

Neither command opens the test CSV. The old `--skip-test-eval` option remains
accepted for compatibility but is unnecessary.

| Family | Implemented upgrades |
|---|---|
| LR | Configurable regularization/class weighting; word + character TF-IDF |
| DNN | Optional 512-wide residual blocks, LayerNorm and GELU |
| RNN/BiRNN, GRU/BiGRU, LSTM/BiLSTM | Trainable embeddings, separate PAD/UNK, longer inputs, compact classifier |
| Attention | Attention + mean + max pooling |
| Custom Transformer | Independent attention initialization, input normalization, pooling/depth/width search |
| DistilBERT | Full fine-tuning, CLS + mean pooling, encoder/head learning rates, warmup/decay |

Shared training supports gradient accumulation, AMP, AdamW parameter groups and
validation-macro-F1 checkpoint selection. Legacy vector-input checkpoints remain
loadable; new sequence checkpoints use `word2vec_ids`.

## 2. Export and check CPU suitability ? validation data only

```powershell
$selection = Get-Content results/improvement/selection.json -Raw | ConvertFrom-Json
python main.py export --selection results/improvement/selection.json --output deploy/vercel-api/model
python main.py benchmark --model deploy/vercel-api/model --reference $selection.model_dir
```

The benchmark is for you to run; it evaluates the saved validation partition
from the training CSV. Export removes optimizer state and duplicate base-model
weights. Use an empty output directory to protect previous releases.

CPU gates: peak process RAM below 1.5 GB, warm p95 prediction below five seconds,
initialization plus prediction below 60 seconds, ten-item batch below 60 seconds,
and validation macro-F1 loss no greater than 0.1 percentage point. These local
measurements do not guarantee Vercel performance.

If the FP32 export fails, create a separate INT8 candidate:

```powershell
python main.py export --selection results/improvement/selection.json --quantize --output deploy/vercel-int8/model
python main.py benchmark --model deploy/vercel-int8/model --reference $selection.model_dir --output results/deployment_benchmark_int8.json
```

Only proceed with a passing candidate. For INT8, substitute `deploy/vercel-int8`
for `deploy/vercel-api` below. No automatic LR fallback or paid upgrade occurs.

## 3. Final test evaluation ? freeze choices first

```powershell
python main.py evaluate-selected --selection results/improvement/selection.json --data data/Test_data.csv --output results/final_test
python main.py evaluate --model deploy/vercel-api/model --data data/Test_data.csv --output results/deployed_test --device cpu
```

The first command evaluates the selected checkpoint from each family. The
second measures the exact serving artifact. Evaluation never changes the
validation-selected winner. Do not tune again against these test scores.

Reports include macro F1, accuracy, per-class metrics, confusion matrices,
error analysis and a bootstrap 95% macro-F1 confidence interval. The serving
model card links scores to the artifact and dataset hashes.

## 4. Prepare the API and publish its model artifact

```powershell
python main.py prepare-vercel --output deploy/vercel-api
hf auth login
python scripts/publish_model.py --repo YOUR_USERNAME/news-topic-classifier --folder deploy/vercel-api/model
```

The publish command creates a private Hugging Face model repository and prints
`MODEL_REPO_ID` and the immutable `MODEL_REVISION`. This is model storage;
inference runs on Vercel CPU. No Space or paid compute is required. Preparation
pins serialization-library versions to the local environment.

Create the API Vercel project:

```powershell
Set-Location deploy/vercel-api
npx vercel link
npx vercel env add API_KEY
npx vercel env add MODEL_REPO_ID
npx vercel env add MODEL_REVISION
npx vercel env add HF_TOKEN
npx vercel env add VERCEL_SUPPORT_LARGE_FUNCTIONS
npx vercel
```

Add variables for both Preview and Production when prompted:

| Variable | Value |
|---|---|
| `API_KEY` | A long random secret; reuse it as the frontend's `INFERENCE_API_KEY` |
| `MODEL_REPO_ID` | Repository printed by the upload command |
| `MODEL_REVISION` | Full 40-character commit SHA printed by that command |
| `HF_TOKEN` | Read token for the private model repository |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | `1` |

The build downloads and verifies model files. Requests do not download weights.
Use Python 3.12 and the included entry point/build configuration. Vercel Hobby
has 2 GB memory; eligible large-functions beta bundles may be up to 5 GB.
See [Vercel's limits](https://vercel.com/docs/functions/limitations). Stay on Hobby;
no paid hosting is assumed.

Check authenticated `/ready`, predictions, batches, cold starts and logs on the
preview before production:

```powershell
npx vercel --prod
Set-Location ../..
```

## 5. Run and deploy the website

Start the local API in another terminal:

```powershell
$env:MODEL_DIR = "deploy/vercel-api/model"
$env:API_KEY = "YOUR_LOCAL_SECRET"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Start the website:

```powershell
Set-Location web
Copy-Item .env.example .env.local
npm install
npm run dev
```

Edit `.env.local` to match the local API secret. Open `http://localhost:3000`.
Do not use `NEXT_PUBLIC_` for credentials.

For deployment, from `web/`:

```powershell
npx vercel link
npx vercel env add INFERENCE_API_URL
npx vercel env add INFERENCE_API_KEY
npx vercel
```

Set the API production URL and secret for both Preview and Production. If the
API has Vercel deployment protection, also add `INFERENCE_VERCEL_BYPASS_SECRET`.
The browser calls same-origin Next.js routes; credentials stay on the server.
After checking the preview:

```powershell
npx vercel --prod
```

The website includes single/batch predictions, topic probabilities, separate
validation/test metrics, artifact identity and a read-only family comparison.
Retain the prior artifact revision and Vercel deployment for rollback.

## Optional checks for you to run

From the project root, in the Python environment:

```powershell
python -m pytest tests -q
```

Archived Streamlit tests skip if Streamlit is absent. New regression coverage
includes training without a test CSV, conflicting labels, PAD/UNK separation,
padding invariance, independent attention initialization and artifact loading.

From `web/`:

```powershell
npm run typecheck
npm run build
```

No training, evaluation, test, build or deployment commands above were executed
by the implementation agent.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and model-loaded flag |
| `GET /ready` | Load/check pinned model; return 503 on failure |
| `POST /predict` | One headline, at most 5,000 characters |
| `POST /batch_predict` | Up to 100 headlines; website sends chunks of ten |
| `GET /model/info` | Metadata and metrics for the actual loaded artifact |
| `GET /model/version` | API version and artifact hash |
| `GET /models` | Read-only family comparison |

All endpoints except liveness require `X-API-Key` when configured. An API key
is mandatory on Vercel. One model is loaded per instance; overlapping inference
receives 429 with a retry hint.

## Attribution

Educational CSE 440 portfolio project by Nowshin, 2026. Uses Google News
Word2Vec and DistilBERT pretrained resources. See `LICENSE`.
