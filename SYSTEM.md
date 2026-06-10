# Face-Guard — Facial Recognition System

## What It Is

A production-grade cybersecurity facial recognition API. Registered users are enrolled with a face image; subsequent requests verify identity (1:1) or identify a person from a crowd (1:N). Every request passes through liveness detection so a printed photo or video replay cannot defeat it.

---

## Architecture

```
Client
  │
  │ HTTPS  (JWT Bearer token required on all /verify /identify /register)
  ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI  (app/server.py)                                   │
│                                                             │
│  Middleware stack:                                          │
│    SecurityHeadersMiddleware  — HSTS, CSP, X-Frame-Options  │
│    RateLimitMiddleware        — 120 req/min per IP          │
│    CORSMiddleware                                           │
│                                                             │
│  Routes:                                                    │
│    POST /api/v1/auth/token      → JWT                       │
│    GET  /api/v1/health          → status                    │
│    POST /api/v1/register        → enrol face                │
│    POST /api/v1/verify          → 1:1 match                 │
│    POST /api/v1/identify        → 1:N search                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Face Service  (app/services/face_service.py)               │
│                                                             │
│  Pipeline for every image:                                  │
│    1. ImageProcessor.process_image()                        │
│       ├─ detect_and_crop_face()   InsightFace SCRFD        │
│       ├─ check_quality()          blur + contrast           │
│       └─ low_light_enhance()      Zero-DCE (if needed)      │
│    2. Liveness.check_liveness()   DeepFace anti-spoofing    │
│    3. FaceAnalysis.get()          InsightFace embedding     │
│       └─ normed_embedding  512-dim L2-normalised vector     │
└────────┬───────────────────┬────────────────────────────────┘
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌───────────────────────────────────────┐
│  MySQL          │  │  Milvus / Zilliz Cloud                │
│  (app/db/pool)  │  │  (app/db/vector)                      │
│                 │  │                                        │
│  users          │  │  Collection: surveillance_faces        │
│  ├─ embedding   │  │  ├─ user_id  (primary key)            │
│  │  (AES-GCM    │  │  ├─ embedding  FLOAT_VECTOR dim=512   │
│  │   encrypted) │  │  ├─ det_score                        │
│  ├─ nonce       │  │  ├─ model_version                    │
│  ├─ hmac        │  │  └─ registration_time                 │
│  ├─ det_score   │  │                                        │
│  └─ reg_time    │  │  Index: HNSW  metric=IP (cosine)       │
│                 │  └───────────────────────────────────────┘
│  authorization_logs                                         │
│  failed_attempts (brute-force lockout)                      │
└─────────────────┘
```

---

## ML Pipeline Detail

### 1. Image Quality & Pre-processing
Module: `app/image_processing.py` — `ImageProcessor`

| Check | Method | Threshold |
|---|---|---|
| Minimum size | shape check | 112×112 px |
| Blur | Tenengrad (Sobel gradient variance) | variance > 40 |
| Contrast | Histogram CDF range + Michelson contrast | range ≥ 20, MC ≥ 0.1 |
| Low light | Mean brightness | < 0.3 → enhance |

Low-light enhancement uses **Zero-DCE** (Deep Curve Estimation network, 8 iterative curve applications). Weights loaded from `models/Epoch99.pth`.

After quality pass → face is cropped (10% padding around detected bbox).

### 2. Liveness Detection
Module: `app/liveness.py` — `Liveness.check_liveness()`

Uses **DeepFace** with `anti_spoofing=True`, `detector_backend='opencv'`. Returns:
- `True` → real face, continue
- `False` → spoof detected → `LivenessFailed` exception → 422 response

### 3. Face Embedding
Module: `app/services/face_service.py` → InsightFace `FaceAnalysis`

Model: **antelopev2** (ArcFace variant, ResNet100 backbone)
- Detection: `scrfd_10g_bnkps.onnx` — SCRFD detector, det_size=384×384
- Recognition: `glintr100.onnx` — 512-dim L2-normalised embedding
- Runs on GPU (CUDA) if `INSIGHTFACE_GPU=true`

Similarity metric: **Inner Product** (= cosine similarity on L2-normalised vectors).
Threshold: configurable via `SIMILARITY_THRESHOLD` (default 0.5).

### 4. Encryption
Module: `app/encrypt.py` — `encrypt`

| Component | Algorithm |
|---|---|
| Embedding storage | AES-256-GCM (AEAD) |
| Integrity check | HMAC-SHA256 |
| User ID anonymisation | BLAKE2b (16-byte digest) |

MySQL stores `(ciphertext, nonce, hmac)`. Zilliz stores plaintext embeddings for vector search.

---

## API Reference

### Authentication
All endpoints except `/health` require a JWT in the `Authorization: Bearer <token>` header.

**Get token:**
```
POST /api/v1/auth/token
Content-Type: application/json

{"api_key": "<your-api-key>"}

→ {"access_token": "eyJ...", "token_type": "bearer"}
```
Token expires in 30 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`).

---

### Register a Face
```
POST /api/v1/register
Authorization: Bearer <token>
Content-Type: multipart/form-data

user_id=alice
image=<file>

→ 201  {"registered": true, "user_id": "alice", "det_score": 0.97}
→ 409  user already registered
→ 422  no face / liveness failed / image quality too low
```

---

### 1:1 Verify (Is this alice?)
```
POST /api/v1/verify
Authorization: Bearer <token>
Content-Type: multipart/form-data

user_id=alice
image=<file>

→ 200  {"authorized": true,  "similarity": 0.82, "confidence": "82.0%", "user_id": "alice"}
→ 200  {"authorized": false, "similarity": 0.31, "confidence": "31.0%", "user_id": "alice"}
→ 404  user not registered
→ 423  account locked (too many failed attempts)
→ 422  liveness / face detection failed
```
After `MAX_FAILED_ATTEMPTS` (default 5) denied attempts, the account is locked for `LOCKOUT_SECONDS` (default 300s).

---

### 1:N Identify (Who is this?)
```
POST /api/v1/identify
Authorization: Bearer <token>
Content-Type: multipart/form-data

image=<file>

→ 200  {"identified": true,  "user_id": "alice", "score": 0.79, "confidence": "79.0%"}
→ 200  {"identified": false, "user_id": null,    "score": 0.21, "confidence": "21.0%"}
→ 422  liveness / face detection failed
```

---

### Health Check
```
GET /api/v1/health  (no auth required)

→ 200  {"mysql": "ok", "milvus": "ok (entities=12)", "insightface": "ok", ...}
→ 503  one or more subsystems degraded
```

---

## Security Design

| Concern | Measure |
|---|---|
| Auth | API key → JWT (30min expiry). No unauthenticated access to biometric endpoints. |
| Transport | HSTS header enforced. TLS must be terminated upstream (nginx/Vast.ai). |
| Brute force | Per-user lockout after 5 failed verify attempts, 5-minute cooldown. |
| Biometric storage | Embeddings encrypted at rest (AES-256-GCM + HMAC-SHA256). Plaintext only in memory and Zilliz (vector search requires plaintext vectors). |
| Liveness | Every request checked by DeepFace anti-spoofing before embedding extraction. |
| Input validation | `user_id` validated by regex `^[A-Za-z0-9_\-]{1,64}$`. Image capped at 10MB, only JPEG/PNG/WebP accepted. |
| Rate limiting | 120 req/min per IP (sliding window, in-process). |
| Error messages | Internal details never returned. Generic messages only. |
| Logging | Structured JSON logs. IP address and similarity scores stored in `authorization_logs`. Sensitive data (embeddings, keys) never logged. |

---

## Email Feedback (Operational Alerts)

The `EmailFeedback` class (`app/email_feedback.py`) is an **operational alerting system**.

It sends HTML emails via SMTP (Gmail by default) when:
- Models initialise successfully or fail
- Encryption errors occur
- Face processing fails unexpectedly

Configuration in `.env`:
```
SENDER_EMAIL=youraddr@gmail.com
SENDER_PASSWORD=your-app-password
RECIPIENT_EMAIL=alerts@yourcompany.com
```

---

## Template Adaptation (Feedback Training Loop)

The system has a **built-in template adaptation loop** in `app/services/face_service.py`.

### How it works

Every time a user is verified with **high confidence** (similarity ≥ 0.72), their stored face template is automatically updated:

```
new_template = 0.90 × old_template + 0.10 × new_embedding
new_template = new_template / ‖new_template‖   (re-normalise to unit length)
```

This is an **Exponential Moving Average (EMA)** blend. The stored template gradually shifts to cover:
- Different lighting conditions
- Slight angle changes
- Expression variations
- Ageing over time

The neural network weights (`glintr100.onnx`) are **not** retrained. Only the stored per-user template vector is updated. This is safe, auditable, and requires no GPU training budget.

### Thresholds
| Parameter | Value | Meaning |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.5 | Minimum to accept a match |
| `_ADAPT_THRESHOLD` | 0.72 | Minimum to trigger template update |
| `_ADAPT_ALPHA` | 0.90 | Weight of old template (conservative) |

High adapt threshold (0.72 >> 0.5) means only clearly genuine matches update the template, protecting against adversarial drift.

### Storage path
1. New blended embedding → **re-encrypted** (AES-GCM + HMAC) → MySQL `users.embedding` updated
2. New embedding → **Milvus** `upsert_face()` → old vector replaced with blended vector
3. Logged to `authorization_logs` as usual

---

## Configuration Reference

All settings are in `.env` (loaded by Pydantic Settings):

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | required | Master secret for token exchange |
| `MYSQL_HOST` | required | MySQL server address |
| `MYSQL_PORT` | 3306 | |
| `MYSQL_USER` | required | |
| `MYSQL_PASSWORD` | required | |
| `MYSQL_DATABASE` | required | |
| `ZILLIZ_URI` | required | Zilliz Cloud cluster URI |
| `ZILLIZ_TOKEN` | required | Zilliz API token |
| `AES_KEY` | required | 64 hex chars (32 bytes) |
| `HMAC_KEY` | required | ≥64 hex chars (≥32 bytes) |
| `ANONYMIZATION_KEY` | required | 64 hex chars (32 bytes) |
| `SIMILARITY_THRESHOLD` | 0.5 | Min cosine similarity to accept a match |
| `MAX_FAILED_ATTEMPTS` | 5 | Lockout trigger count |
| `LOCKOUT_SECONDS` | 300 | Lockout duration |
| `INSIGHTFACE_GPU` | true | Use CUDA for inference |
| `INSIGHTFACE_MODEL` | antelopev2 | InsightFace model pack |
| `MODEL_CACHE_DIR` | ./models/.insightface | |
| `DCE_MODEL_PATH` | ./models/Epoch99.pth | Zero-DCE weights |
| `ENV` | production | Disables /docs in production |
| `WORKERS` | 1 | Keep 1 — models are not fork-safe |

---

## Deployment (Vast.ai)

**Current instance:** `ssh -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p 47331 root@108.55.118.247`

**Start/restart API:**
```bash
pkill -f "uvicorn app.server" 2>/dev/null || true
cd /workspace/face-guard
nohup python3 -m uvicorn app.server:app \
  --host 0.0.0.0 --port 8000 --workers 1 \
  > /workspace/api.log 2>&1 &
```

**Check logs:**
```bash
tail -f /workspace/api.log
```

**Future deployments (from local machine):**
```bash
bash deploy.sh <SSH_PORT> <HOST>
```

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| MySQL | Running | Schema auto-created on startup |
| Milvus (Zilliz) | Resume in progress | DNS propagating after cluster resume |
| InsightFace (antelopev2) | Loaded on GPU | CUDA confirmed |
| Zero-DCE | Loaded | `models/Epoch99.pth` |
| DeepFace liveness | Loaded | TF 2.20 + tf-keras |
| AES-GCM encryption | Ready | Keys validated at startup |
| Auth (JWT) | Working | API key verified, 30-min tokens |

**To reconnect Milvus** after the cluster finishes resuming:
```bash
# On Vast.ai:
pkill -f "uvicorn app.server"
# ... then restart with the command above
```
If the Zilliz cluster URI changed after resuming, update `ZILLIZ_URI` and `ZILLIZ_TOKEN` in `/workspace/face-guard/.env` first.
