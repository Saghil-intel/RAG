# Transcript retrieval (Docling + BGE-M3 + Chroma + hybrid + reranker) - no LLM

    pip install -r requirements.txt
    python -m rag.cli ingest Transcript.pdf
    python -m rag.cli search "how will Intel fix the inference latency?" -k 8 --hops 1
    uvicorn rag.api:app --reload      # POST /ingest, POST /search

search = BGE-M3 dense (Chroma) + BGE-M3 sparse + BM25 -> RRF -> optional feedback hop
-> bge-reranker-v2-m3 -> neighbour-turn expansion. Settings in rag/config.py.

## LLM agent (local, Ollama + qwen3.5:4b-q4_K_M)

    ollama pull qwen3.5:4b-q4_K_M
    python -m rag.cli ask "How does Intel's proposal address Assessli's inference bottleneck?" --trace
    # POST /ask {"question": "..."}

Env: RAG_LLM, OLLAMA_BASE_URL, RAG_NUM_CTX (16384), RAG_NUM_PREDICT (1024),
RAG_THINKING=1 (enable thinking), RAG_FAITH_CHECK=0 (skip the LLM faithfulness judge).
graph.py = LangGraph loop (understand -> decompose -> retrieve -> evaluate -> generate -> validate).
`search` stays LLM-free.

## Transcript + questionnaire (two sources, one vector store)

Source priority on conflicts: **1) your prompt  2) questionnaire  3) transcript.**
Use the SAME `meeting_id` for a transcript and its questionnaire.

    python -m rag.cli ingest Transcript.pdf --questionnaire Technical_Questionnaire.pdf --meeting-id intel-assessli
    python -m rag.cli ingest-questionnaire Technical_Questionnaire.pdf --meeting-id intel-assessli
    python -m rag.cli ask "Which Python version does Assessli use?" --meeting intel-assessli --trace

Swagger (`uvicorn rag.api:app --reload`, then http://localhost:8000/docs):
`POST /run` (prompt + optional transcript + optional questionnaire, one call),
`POST /ingest/transcript`, `POST /ingest/questionnaire`, `POST /ask`, `POST /search`, `GET /meetings`.

Questionnaire: Docling table -> one chunk per row (column 1 = question, column 2 = answer).
Chroma metadata per chunk: `source` (transcript|questionnaire), `priority` (3|2), `meeting_id`, `row_index`.
Retrieval: hybrid search + rerank run separately per source (own quota), then merged with a soft
priority weight; the answer prompt applies the conflict rule and cites [Q#] rows and [E#] turns.
