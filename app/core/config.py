import secrets
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # ── API Security ──────────────────────────────────────────────────────────
    SECRET_KEY: str = secrets.token_hex(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    API_KEY: str  # Must be set — guards every endpoint

    # ── Database ──────────────────────────────────────────────────────────────
    MYSQL_HOST: str
    MYSQL_PORT: int = 3306
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_DATABASE: str
    MYSQL_POOL_SIZE: int = 5
    MYSQL_POOL_NAME: str = "facerec_pool"

    # ── Milvus / Zilliz ───────────────────────────────────────────────────────
    ZILLIZ_URI: str
    ZILLIZ_TOKEN: str

    # ── Encryption ────────────────────────────────────────────────────────────
    AES_KEY: str          # 64 hex chars → 32 bytes
    HMAC_KEY: str         # ≥ 64 hex chars
    ANONYMIZATION_KEY: str  # 64 hex chars → 32 bytes

    # ── Email alerts ─────────────────────────────────────────────────────────
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SENDER_EMAIL: Optional[str] = None
    SENDER_PASSWORD: Optional[str] = None
    RECIPIENT_EMAIL: Optional[str] = None

    # ── Face recognition ─────────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.5
    MAX_FAILED_ATTEMPTS: int = 5
    LOCKOUT_SECONDS: int = 300

    # ── Rate limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_RECOGNIZE: str = "30/minute"
    RATE_LIMIT_REGISTER: str = "10/minute"
    RATE_LIMIT_GLOBAL: str = "120/minute"

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Model ─────────────────────────────────────────────────────────────────
    INSIGHTFACE_MODEL: str = "antelopev2"
    MODEL_CACHE_DIR: str = "./models/.insightface"
    DCE_MODEL_PATH: str = "./models/Epoch99.pth"
    INSIGHTFACE_GPU: bool = True        # Set False for CPU-only nodes
    INSIGHTFACE_DET_SIZE: int = 384

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1                    # Keep 1; models aren't fork-safe
    ENV: str = "production"

    @field_validator("AES_KEY")
    @classmethod
    def validate_aes_key(cls, v: str) -> str:
        assert len(bytes.fromhex(v)) in (16, 24, 32), \
            "AES_KEY must decode to 16, 24, or 32 bytes"
        return v

    @field_validator("HMAC_KEY")
    @classmethod
    def validate_hmac_key(cls, v: str) -> str:
        assert len(bytes.fromhex(v)) >= 32, \
            "HMAC_KEY must decode to at least 32 bytes"
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
