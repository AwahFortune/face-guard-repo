"""
Milvus / Zilliz vector database wrapper.
Manages the single connection and surveillance_faces collection.
"""
import logging
import time

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from ..core.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "surveillance_faces"
_collection: Collection | None = None


def init_milvus() -> Collection:
    global _collection

    connections.connect(
        alias="default",
        uri=settings.ZILLIZ_URI,
        token=settings.ZILLIZ_TOKEN,
        secure=True,
    )

    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="user_id", dtype=DataType.VARCHAR,
                        max_length=64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="det_score", dtype=DataType.FLOAT),
            FieldSchema(name="model_version", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="registration_time", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="Surveillance face templates")
        col = Collection(COLLECTION_NAME, schema)
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW",
                          "metric_type": "IP",
                          "params": {"M": 16, "efConstruction": 50}},
        )
        logger.info("Created new Milvus collection '%s'", COLLECTION_NAME)
    else:
        col = Collection(COLLECTION_NAME)
        logger.info("Loaded existing Milvus collection '%s'", COLLECTION_NAME)

    col.load()
    _collection = col
    return _collection


def get_collection() -> Collection:
    if _collection is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail={"error": "MILVUS_UNAVAILABLE",
                    "message": "Vector database is not connected"},
        )
    return _collection


def upsert_face(user_id: str, embedding: list, det_score: float,
                model_version: str) -> None:
    col = get_collection()
    # Delete existing entry for the same user_id before inserting
    col.delete(f'user_id == "{user_id}"')
    col.insert([[user_id], [embedding], [det_score],
                [model_version], [int(time.time())]])
    col.flush()


def search_face(embedding: list, top_k: int = 1) -> list:
    """Return list of (user_id, score) for the top-k matches."""
    col = get_collection()
    results = col.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["user_id"],
    )
    hits = []
    for r in results[0]:
        hits.append({"user_id": r.entity.get("user_id"), "score": float(r.score)})
    return hits


def delete_face(user_id: str) -> None:
    col = get_collection()
    col.delete(f'user_id == "{user_id}"')
    col.flush()
