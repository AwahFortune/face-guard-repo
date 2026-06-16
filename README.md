# 🛡️ FaceGuard AI — Facial Recognition System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-0D6EFD?style=for-the-badge)

**FaceGuard AI** is a production-grade cybersecurity facial recognition API. It provides a robust, highly secure architecture for user enrollment (1:1 Verification) and crowd identification (1:N Identification). 

Every request passes through strict **liveness detection** to ensure that printed photos or video replays cannot defeat the system, making it suitable for high-security access control and surveillance.

---

## 🏗️ System Architecture

FaceGuard AI is built with scalability and security at its core:

1. **FastAPI Gateway:** Handles all incoming HTTPS requests with security middleware (HSTS, CSP, Rate Limiting, CORS).
2. **Face Service Pipeline:**
   - **Image Processor:** Validates size, blur, and contrast. Enhances low-light images using **Zero-DCE**.
   - **Liveness Detection:** DeepFace anti-spoofing mechanism to prevent bypass attacks.
   - **Embedding Extraction:** Uses InsightFace (antelopev2) to generate 512-dimensional L2-normalised vectors.
3. **Dual Database Storage:**
   - **MySQL:** Stores AES-256-GCM encrypted embeddings at rest with HMAC integrity checks.
   - **Zilliz Cloud / Milvus:** Stores plaintext vectors for ultra-fast Cosine Similarity (Inner Product) search.

## 🚀 Key Features

- **Biometric Encryption at Rest:** All sensitive biometric data stored in MySQL is encrypted (AES-256-GCM) to prevent database dumps from leaking facial features.
- **Template Adaptation (Feedback Loop):** High-confidence verifications automatically blend the new embedding with the stored template using an Exponential Moving Average (EMA). This allows the system to naturally adapt to aging, expression variations, and lighting changes without expensive GPU retraining.
- **Low-Light Enhancement:** Implements Deep Curve Estimation (Zero-DCE) for scenarios with poor lighting.
- **Robust Security:** JWT Bearer tokens, brute-force lockouts, API rate limits (120 req/min), and strict input validation.

## 🛠️ Setup & Deployment

### Prerequisites
- Python 3.10+
- MySQL Server
- Milvus/Zilliz Cloud account
- Optional: CUDA-enabled GPU for faster inference (`INSIGHTFACE_GPU=true`)

### Installation & Model Download

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your environment variables inside `.env` (copy from `.env.example`):
   ```env
   API_KEY=your_master_secret
   MYSQL_HOST=localhost
   ZILLIZ_URI=your_zilliz_cluster_uri
   AES_KEY=your_32_byte_hex_key
   ```
3. **Download Models:**
   - The InsightFace models (`antelopev2`) are large (~200MB+). They are **not** included in this repository.
   - When you start the application for the first time, InsightFace will automatically download the required `.onnx` models into the `models/.insightface/` directory. Ensure you have a stable internet connection.
   - *If you are deploying to an offline environment, you must pre-download the `antelopev2.zip` model pack from the InsightFace GitHub releases and extract it to `models/.insightface/models/antelopev2/`.*

4. Start the application:
   ```bash
   uvicorn app.server:app --host 0.0.0.0 --port 8000
   ```

## 📖 API Documentation

FaceGuard provides Swagger UI out of the box when running locally on your device.

- `POST /api/v1/auth/token`: Retrieve a JWT token.
- `POST /api/v1/register`: Enroll a new user face.
- `POST /api/v1/verify`: 1:1 Identity Verification.
- `POST /api/v1/identify`: 1:N Identity Search.

---
