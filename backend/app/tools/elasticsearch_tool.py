# backend/app/tools/elasticsearch_tool.py
# ─────────────────────────────────────────────────────────────────────────────
# Elasticsearch LangChain tools — BGE-M3 dense embeddings + kNN retrieval.
#
# Uses BAAI/bge-m3 as the local embedding model:
#   - Runs entirely on-device (no API key needed)
#   - Multilingual: 100+ languages
#   - 1024-dim dense vectors stored in ES kNN index
#   - Model is downloaded from HuggingFace on first boot (~570 MB)
#
# Config (backend/.env):
#   ES_HOST         = http://localhost:9200
#   ES_USERNAME     = elastic
#   ES_PASSWORD     = changeme
#   ES_VECTOR_INDEX = nextbrick-vectors
#   EMBEDDING_MODEL = BAAI/bge-m3       ← default
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import structlog
from typing import Optional
from langchain_core.tools import tool

log = structlog.get_logger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_embeddings = None   # type: Optional[object]
_vector_store = None  # type: Optional[object]


def _get_embeddings():
    """
    Return a singleton BGE-M3 embedding model.
    Downloaded from HuggingFace on first call (~570 MB), cached locally after that.
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    from app.config import settings

    model_name = getattr(settings, "embedding_model", "BAAI/bge-m3")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},          # switch to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity
        )
        log.info("bge_m3.embedder_ready", model=model_name)

    except Exception as e:
        log.error("bge_m3.embedder_failed", model=model_name, error=str(e))
        _embeddings = None

    return _embeddings


def _get_vector_store():
    """
    Return a singleton ElasticsearchStore backed by BGE-M3 dense vectors.
    Uses kNN approximate nearest-neighbour search for fast retrieval.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    from app.config import settings

    embeddings = _get_embeddings()
    if embeddings is None:
        log.error("elasticsearch_tool.store_skip", reason="BGE-M3 embedder unavailable")
        return None

    try:
        from langchain_elasticsearch import ElasticsearchStore
        from elasticsearch import Elasticsearch

        es_kwargs: dict = {"hosts": [settings.es_host]}
        if settings.es_username and settings.es_password:
            es_kwargs["basic_auth"] = (settings.es_username, settings.es_password)

        es_client = Elasticsearch(**es_kwargs)

        _vector_store = ElasticsearchStore(
            es_connection=es_client,
            index_name=settings.es_vector_index,
            embedding=embeddings,
        )
        log.info(
            "elasticsearch_tool.store_ready",
            strategy="BGE-M3 dense kNN",
            host=settings.es_host,
            index=settings.es_vector_index,
        )
        _seed_if_empty(_vector_store)

    except Exception as e:
        log.error("elasticsearch_tool.store_init_failed", error=str(e))
        _vector_store = None

    return _vector_store


def _seed_if_empty(store) -> None:
    """
    Ingest sample documents on first boot so the index is never empty.
    Uses stable IDs — re-running is safe (no duplicates).
    """
    from langchain_core.documents import Document

    seed_docs = [
        Document(
            page_content="IoT Sensor v3 installation requires 24V DC power and RS-485 wiring. "
                         "Calibration must be performed with NextCal v2 within 72 hours of install. "
                         "Supports Modbus RTU and MQTT protocols.",
            metadata={"source": "product-manual", "title": "IoT Sensor v3 Manual"},
        ),
        Document(
            page_content="Customer onboarding takes 5-7 business days: account setup, "
                         "license activation, admin training, go-live sign-off with the CS team.",
            metadata={"source": "confluence", "title": "Customer Onboarding Process"},
        ),
        Document(
            page_content="Calibration certificates are issued annually. "
                         "Log into the Nextbrick portal → Assets → Calibration → Request Certificate. "
                         "PDF delivered by email within 3 business days.",
            metadata={"source": "confluence", "title": "Calibration Certificate FAQ"},
        ),
        Document(
            page_content="Support SLA: P1 Critical 2-hour response, P2 High 8-hour, "
                         "P3 Medium 2 business days, P4 Low 5 business days.",
            metadata={"source": "support-policy", "title": "SLA Policy"},
        ),
        Document(
            page_content="APAC regional pricing includes a 5% surcharge. "
                         "Volume discounts: 10 units 5%, 50 units 12%, 100+ units 18%. "
                         "Discounts above 25% require Sales VP approval.",
            metadata={"source": "pricing-guide", "title": "Regional Pricing FY2026"},
        ),
    ]
    try:
        ids = [f"seed-bge-{i+1}" for i in range(len(seed_docs))]
        store.add_documents(seed_docs, ids=ids)
        log.info("elasticsearch_tool.seed_docs_ingested", count=len(seed_docs))
    except Exception as e:
        log.debug("elasticsearch_tool.seed_skip", reason=str(e))


# ── LangChain Tools ────────────────────────────────────────────────────────────

@tool
def elasticsearch_semantic_search(query: str, top_k: int = 5) -> list:
    """
    Search indexed documents using BGE-M3 semantic embeddings and kNN retrieval.
    BGE-M3 understands meaning, synonyms, and cross-language queries — not just keywords.
    Use this as the default search tool for any question about products, policies, or processes.
    Returns top matching passages with source and title metadata.
    """
    log.info("elasticsearch.semantic_search", query=query, top_k=top_k)
    store = _get_vector_store()
    if store is None:
        return [{"error": "Elasticsearch / BGE-M3 not ready. Check ES_HOST credentials and that the model downloaded correctly.", "query": query}]

    try:
        docs = store.similarity_search(query, k=top_k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "title": doc.metadata.get("title", ""),
                "retrieval_method": "BGE-M3 kNN",
            }
            for doc in docs
        ]
    except Exception as e:
        log.error("elasticsearch.semantic_search.error", error=str(e))
        return [{"error": str(e), "query": query}]


@tool
def elasticsearch_ingest_document(title: str, content: str, source: str = "manual") -> dict:
    """
    Index a new document into Elasticsearch using BGE-M3 embeddings.
    The content is chunked, embedded with BGE-M3, and stored for semantic retrieval.
    Use when the user wants to add a new knowledge article, guide, policy, or FAQ.
    Returns number of chunks indexed.
    """
    log.info("elasticsearch.ingest", title=title, source=source)
    store = _get_vector_store()
    if store is None:
        return {"error": "Elasticsearch not available.", "indexed": False}

    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents(
            [content],
            metadatas=[{"title": title, "source": source}],
        )
        store.add_documents(chunks)
        log.info("elasticsearch.ingest.done", title=title, chunks=len(chunks))
        return {
            "indexed": True,
            "title": title,
            "source": source,
            "chunks_stored": len(chunks),
            "retrieval_method": "BGE-M3 kNN",
        }
    except Exception as e:
        log.error("elasticsearch.ingest.error", error=str(e))
        return {"error": str(e), "indexed": False}
