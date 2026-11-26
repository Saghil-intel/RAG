
# config.py

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # storage
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "./data"))
    collection: str = "transcript_chunks"

    # models
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    use_fp16: bool = os.getenv("RAG_FP16", "0") == "1"

    # local LLM via LM Studio
    llm_model: str = os.getenv("RAG_LLM", "qwen2.5-vl-3b-instruct")
    lmstudio_url: str = os.getenv(
        "LMSTUDIO_BASE_URL", "http://localhost:1234/v1"
    )
    lmstudio_api_key: str = os.getenv(
        "LMSTUDIO_API_KEY", "lm-studio"
    )

    llm_num_ctx: int = int(os.getenv("RAG_NUM_CTX", "16384"))
    llm_num_predict: int = int(os.getenv("RAG_NUM_PREDICT", "1024"))
    llm_thinking: bool = os.getenv("RAG_THINKING", "0") == "1"
    llm_faithfulness_check: bool = os.getenv("RAG_FAITH_CHECK", "1") == "1"

    # source priority
    weight_questionnaire: float = 1.0
    weight_transcript: float = 0.8
    questionnaire_k: int = 4
    min_rerank_score: float = 0.02
    max_evidence_rows: int = 8

    # chunking
    short_turn_words: int = 8
    min_chunk_tokens: int = 80
    target_chunk_tokens: int = 220
    max_chunk_tokens: int = 320
    gap_seconds: int = 90

    # retrieval
    dense_k: int = 30
    sparse_k: int = 30
    bm25_k: int = 30
    rrf_k: int = 60
    fused_k: int = 40
    rerank_k: int = 6
    neighbor_window: int = 1

    # agent loop bounds
    max_iterations: int = 3
    max_retrieval_calls: int = 8
    max_subqueries: int = 4
    max_evidence_turns: int = 24
    max_revisions: int = 1


SOURCE_PRIORITY = {
    "user_prompt": 1,
    "questionnaire": 2,
    "transcript": 3
}

SETTINGS = Settings()
