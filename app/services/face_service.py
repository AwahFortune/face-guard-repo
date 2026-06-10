"""
Face recognition service — orchestrates all ML pipeline steps.
All ML objects are module-level singletons, initialised once at startup.
"""
import logging
import time
from typing import Optional

import numpy as np

from ..core.config import settings
from ..core.exceptions import (
    AccountLocked, FaceNotDetected, ImageQualityError,
    LivenessFailed, UserAlreadyExists, UserNotFound,
)
from ..db.pool import (
    clear_failed_attempts, get_conn,
    get_failed_attempts, record_failed_attempt,
)
from ..db.vector import search_face, upsert_face

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_face_analysis = None    # insightface.app.FaceAnalysis
_image_processor = None  # ImageProcessor
_liveness = None         # Liveness
_encryptor = None        # encrypt


def init_models() -> None:
    """Called once from FastAPI lifespan. Initialises all ML objects."""
    global _face_analysis, _image_processor, _liveness, _encryptor

    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    from ..encrypt import encrypt as Encrypt
    from ..image_processing import ImageProcessor
    from ..liveness import Liveness

    # InsightFace
    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = 4
    session_opts.inter_op_num_threads = 4
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if settings.INSIGHTFACE_GPU
        else ["CPUExecutionProvider"]
    )
    fa = FaceAnalysis(
        name=settings.INSIGHTFACE_MODEL,
        root=settings.MODEL_CACHE_DIR,
        providers=providers,
        session_options=session_opts,
        download=False,
    )
    fa.prepare(
        ctx_id=0 if settings.INSIGHTFACE_GPU else -1,
        det_size=(settings.INSIGHTFACE_DET_SIZE, settings.INSIGHTFACE_DET_SIZE),
    )
    _face_analysis = fa
    logger.info("InsightFace initialised (model=%s)", settings.INSIGHTFACE_MODEL)

    # Image processor
    _image_processor = ImageProcessor(
        app=_face_analysis,
        dce_model_path=settings.DCE_MODEL_PATH,
    )
    logger.info("ImageProcessor initialised")

    # Liveness checker
    _liveness = Liveness()
    logger.info("Liveness checker initialised")

    # Encryption
    enc = Encrypt()
    enc.initialize_security()
    _encryptor = enc
    logger.info("Encryption initialised")


def get_models() -> dict:
    return {
        "insightface": _face_analysis is not None,
        "image_processor": _image_processor is not None,
        "liveness": _liveness is not None,
        "encryptor": _encryptor is not None,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_embedding(img_array: np.ndarray) -> tuple:
    """Process image, check liveness, return (normed_embedding, det_score)."""
    try:
        result = _image_processor.process_image(img_array)
    except ValueError as exc:
        raise ImageQualityError(str(exc)) from exc
    except Exception:
        raise FaceNotDetected()
    # process_image returns (processed_image, faces) on success
    if not isinstance(result, tuple) or len(result) != 2:
        raise FaceNotDetected()
    processed, faces = result
    if not faces:
        raise FaceNotDetected()

    is_live = _liveness.check_liveness(processed)
    if is_live is not True:
        raise LivenessFailed()

    face = faces[0]
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        raise FaceNotDetected()

    return embedding.astype(np.float32), float(face.det_score)


# Minimum similarity to trigger template adaptation (must be clearly above threshold)
_ADAPT_THRESHOLD = 0.72
# Weight of the old template vs new image (0.9 = 90% old, 10% new per verification)
_ADAPT_ALPHA = 0.90


def _adapt_template(user_id: str, new_embedding: np.ndarray, det_score: float) -> None:
    """
    Blend the new embedding into the stored template using an exponential moving
    average.  Only called on high-confidence matches.

    This means the stored template gradually covers different lighting conditions,
    angles, and expressions without retraining the neural network.
    """
    try:
        with get_conn() as (_, cur):
            cur.execute(
                "SELECT embedding, nonce, hmac FROM users WHERE user_id=%s",
                (user_id,)
            )
            row = cur.fetchone()
        if row is None:
            return

        stored_embedding = _encryptor.decrypt_embedding(
            nonce=bytes(row[1]),
            ciphertext=bytes(row[0]),
            mac=bytes(row[2]),
        )

        # EMA blend, then re-normalise to unit length
        blended = _ADAPT_ALPHA * stored_embedding + (1 - _ADAPT_ALPHA) * new_embedding
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm

        nonce, ciphertext, mac = _encryptor.encrypt_embedding(blended.astype(np.float32))
        with get_conn() as (_, cur):
            cur.execute(
                "UPDATE users SET embedding=%s, nonce=%s, hmac=%s WHERE user_id=%s",
                (ciphertext, nonce, mac, user_id),
            )

        upsert_face(user_id, blended.tolist(), det_score, settings.INSIGHTFACE_MODEL)
        logger.info("Template adapted for user=%s (alpha=%.2f)", user_id, _ADAPT_ALPHA)

    except Exception as exc:
        logger.error("Template adaptation failed for user=%s: %s", user_id, exc)


def _log_attempt(
    user_id: str, similarity: float, det_score: float, ip: str, status: str
) -> None:
    try:
        with get_conn() as (_, cur):
            cur.execute(
                """
                INSERT INTO authorization_logs
                    (user_id, similarity, det_score, model_version,
                     ip_address, timestamp, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id, similarity, det_score,
                    settings.INSIGHTFACE_MODEL, ip,
                    int(time.time()), status,
                ),
            )
    except Exception as exc:
        logger.error("Failed to write auth log: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def register_face(user_id: str, img_array: np.ndarray, ip: str) -> dict:
    with get_conn() as (_, cur):
        cur.execute("SELECT 1 FROM users WHERE user_id=%s", (user_id,))
        if cur.fetchone():
            raise UserAlreadyExists()

    embedding, det_score = _extract_embedding(img_array)
    nonce, ciphertext, mac = _encryptor.encrypt_embedding(embedding)
    now = int(time.time())

    with get_conn() as (_, cur):
        cur.execute(
            """
            INSERT INTO users
                (user_id, embedding, nonce, hmac, det_score, model_version, registration_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, ciphertext, nonce, mac, det_score,
             settings.INSIGHTFACE_MODEL, now),
        )

    upsert_face(user_id, embedding.tolist(), det_score, settings.INSIGHTFACE_MODEL)
    logger.info("Registered user=%s det_score=%.3f ip=%s", user_id, det_score, ip)
    return {"det_score": float(det_score)}


def verify_face(user_id: str, img_array: np.ndarray, ip: str) -> dict:
    # Lockout check
    count, locked_until = get_failed_attempts(user_id)
    now = int(time.time())
    if locked_until > now:
        raise AccountLocked(locked_until - now)

    # Existence check
    with get_conn() as (_, cur):
        cur.execute("SELECT 1 FROM users WHERE user_id=%s", (user_id,))
        if not cur.fetchone():
            raise UserNotFound()

    embedding, det_score = _extract_embedding(img_array)

    # Search up to top-5 to find user_id among close matches
    hits = search_face(embedding.tolist(), top_k=5)
    similarity = 0.0
    for h in hits:
        if h["user_id"] == user_id:
            similarity = h["score"]
            break

    authorized = similarity >= settings.SIMILARITY_THRESHOLD
    status = "AUTHORIZED" if authorized else "DENIED"

    if authorized:
        clear_failed_attempts(user_id)
        # Template adaptation: blend new embedding into stored one when
        # the match is high-confidence, improving coverage over time
        if similarity >= _ADAPT_THRESHOLD:
            _adapt_template(user_id, embedding, det_score)
    else:
        record_failed_attempt(user_id)

    _log_attempt(user_id, similarity, det_score, ip, status)
    logger.info("Verify user=%s sim=%.4f status=%s ip=%s", user_id, similarity, status, ip)

    return {"authorized": authorized, "similarity": float(similarity)}


def identify_face(img_array: np.ndarray) -> dict:
    embedding, _ = _extract_embedding(img_array)
    hits = search_face(embedding.tolist(), top_k=1)

    if not hits:
        return {"identified": False, "user_id": None, "score": 0.0}

    top = hits[0]
    identified = top["score"] >= settings.SIMILARITY_THRESHOLD
    return {
        "identified": identified,
        "user_id": top["user_id"] if identified else None,
        "score": float(top["score"]),
    }
