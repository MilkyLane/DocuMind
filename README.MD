# DocuMind

[![CI](https://github.com/MilkyLane/DocuMind/actions/workflows/ci.yml/badge.svg)](https://github.com/MilkyLane/DocuMind/actions/workflows/ci.yml)

An end-to-end document intelligence system. Upload a PDF, image, or scan — DocuMind OCRs it, classifies the document type, extracts structured fields, runs a statistical fraud analysis (on bank statements), and explains why the model made its decision. Every prediction can be rated by a human, and live model metrics are computed from those ratings in real time.

---

## Features

| Feature | Detail |
|---|---|
| **Classification** | 5 classes — Invoice, Resume, Form, Bank Statement, Utility Bill |
| **Field extraction** | Regex + heuristic extractors per class (dates, amounts, names, etc.) |
| **Explainability** | TF-IDF × Logistic Regression coefficients — shows which words drove the prediction |
| **Fraud detection** | Statistical checks on bank statement amounts (round-number bias, outliers, duplicates) |
| **Feedback loop** | Thumbs-up / thumbs-down on every result, with correction dropdown |
| **Live metrics** | Per-class Precision / Recall / F1, confusion matrix — computed from real human corrections |
| **Async processing** | ARQ task queue — uploads return in ~0.5 s, no blocking |
| **Rate limiting** | Per-IP sliding window via Redis |
| **Audit log** | Every prediction stored in PostgreSQL with latency breakdown |

---

## Architecture

```
Browser
  │
  │  POST /predict  (multipart file)
  ▼
FastAPI  ──── Redis (ARQ queue) ────► ARQ Worker
  │                                       │
  │  inline poll (~0.5 s)                 │  1. OCR (Tesseract / pdf2image)
  │◄──────────────────────────────────────┤  2. TF-IDF + Logistic Regression
  │                                       │  3. Field extraction
  │                                       │  4. Fraud analysis (bank stmts)
  │                                       │  5. Write result → Redis
  │                                       │  6. Write log → PostgreSQL
  │
  │  POST /feedback  ──► PostgreSQL (feedback_logs)
  │  GET  /metrics   ◄── PostgreSQL (compute F1 / confusion matrix live)
  │  GET  /logs      ◄── PostgreSQL (document_logs)
  │  POST /explain   ──► ARQ Worker (TF-IDF weight explainer)
```

---

## ML Model

The classifier is a **scikit-learn Pipeline**:

```
TfidfVectorizer  →  CalibratedClassifierCV(LogisticRegression)
```

- **TF-IDF** converts raw OCR text into a sparse feature matrix (top unigrams/bigrams weighted by inverse document frequency)
- **Logistic Regression** with `class_weight="balanced"` handles uneven class sizes
- **CalibratedClassifierCV** (isotonic regression, 5-fold) turns raw scores into proper probabilities — predictions below 40% confidence are rejected as "unsupported"
- Trained on ~500 real + synthetic documents across 5 classes
- Model is serialised with `joblib` to `models/documind_classifier.joblib` and loaded once at worker startup

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.129, Uvicorn |
| Task queue | ARQ (async Redis queue) |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 async + asyncpg |
| Cache / Rate limit | Redis 7 |
| ML | scikit-learn, numpy, joblib |
| OCR | Tesseract 5, pdf2image, Pillow |
| Frontend | Vanilla JS SPA (no framework) |
| Container | Docker (multi-stage, python:3.11-slim) |
| Deployment | Railway (web + worker services) |

---

## Project Structure

```
DocuMind/
├── api/
│   ├── main.py          # FastAPI app — all routes
│   └── worker.py        # ARQ worker — predict + explain tasks
├── src/
│   ├── inference/
│   │   ├── predict.py   # OCR → classify → extract pipeline
│   │   └── explain.py   # TF-IDF × LR coefficient explainer
│   ├── extraction/      # Per-class field extractors
│   │   ├── bank_statement.py
│   │   ├── invoice.py
│   │   ├── resume.py
│   │   ├── form.py
│   │   └── utility.py
│   ├── analysis/
│   │   └── fraud.py     # Statistical fraud checks
│   ├── training/
│   │   ├── train.py     # Model training script
│   │   └── cross_validate.py
│   ├── features/
│   │   └── tfidf.py     # TF-IDF vectoriser config
│   ├── ocr/
│   │   └── ocr_utils.py # Tesseract + pdf2image wrapper
│   ├── db/
│   │   ├── models.py    # SQLAlchemy ORM (DocumentLog, FeedbackLog)
│   │   ├── crud.py      # DB helper functions
│   │   └── database.py  # Async engine + session factory
│   ├── middleware/
│   │   └── rate_limit.py
│   └── utils/
│       └── logging.py
├── frontend/
│   └── index.html       # Full SPA — upload, results, explainability, metrics
├── models/
│   └── documind_classifier.joblib
├── data/
│   ├── processed/       # OCR text files used for training
│   └── raw/             # Original document images/PDFs
├── Dockerfile
├── docker-compose.yml   # Local dev (Postgres + Redis + web + worker)
├── railway.toml         # Railway deployment config
└── requirements.txt
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Upload a document → returns classification + extracted fields |
| `GET` | `/predict/{job_id}` | Poll for result if POST timed out |
| `POST` | `/explain` | Upload a document → returns top TF-IDF features that drove prediction |
| `POST` | `/feedback` | Submit thumbs-up/down + optional correction label |
| `GET` | `/metrics` | Live per-class Precision/Recall/F1 + confusion matrix from feedback |
| `GET` | `/logs` | Paginated document processing history |
| `GET` | `/stats` | Aggregate stats (totals by type/status, avg latencies) |
| `GET` | `/health` | Health check |

---

## Local Development

### Prerequisites
- Docker Desktop

### Run

```bash
git clone https://github.com/MilkyLane/DocuMind.git
cd DocuMind
cp .env.example .env
docker compose up --build
```

Open **http://localhost:8000**

All four services start automatically:
- `web` — FastAPI on port 8000
- `worker` — ARQ worker (listens for jobs)
- `postgres` — PostgreSQL on port 5432
- `redis` — Redis on port 6379

Tables are created automatically on first startup (`Base.metadata.create_all`).

### Retrain the model

```bash
# From inside the web container or with Python env activated:
python -m src.training.train
```

This reads `data/processed/`, trains a new pipeline, and writes `models/documind_classifier.joblib`.

---

## Deployment (Railway)

### 1. Create project
- Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub → select this repo

### 2. Add databases
- **+ New → Database → PostgreSQL**
- **+ New → Database → Redis**

### 3. Add the worker service
- **+ New → GitHub Repo** → select this repo again
- Rename to `worker`
- Set Start Command: `arq api.worker.WorkerSettings`

### 4. Set environment variables (both `web` and `worker`)

| Variable | Value |
|---|---|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` |
| `REDIS_URL` | `${{Redis.REDIS_URL}}` |
| `RATE_LIMIT_REQUESTS` | `60` |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` |
| `JOB_FILE_TTL_SECONDS` | `60` |
| `JOB_RESULT_TTL_SECONDS` | `300` |
| `INLINE_WAIT_SECONDS` | `12.0` |

### 5. Deploy
Railway auto-deploys on every `git push` to `main`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `REDIS_URL` | — | Redis connection string |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per IP per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit rolling window (seconds) |
| `JOB_FILE_TTL_SECONDS` | `60` | How long uploaded bytes live in Redis |
| `JOB_RESULT_TTL_SECONDS` | `300` | How long results live in Redis |
| `INLINE_WAIT_SECONDS` | `12.0` | Max seconds POST /predict waits inline |
