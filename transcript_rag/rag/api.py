import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .config import SETTINGS
from .ingest import ingest_questionnaire, ingest_transcript, slug
from .store import Corpus

app = FastAPI(title="Transcript + Questionnaire RAG",
              description="Source priority on conflicts: 1) your prompt, 2) questionnaire, 3) transcript. "
                          "Use the SAME meeting_id for a transcript and its questionnaire.")

_corpus = _agent = _retriever = None


def corpus() -> Corpus:
    global _corpus
    if _corpus is None:
        _corpus = Corpus(SETTINGS)
    return _corpus


def agent():
    global _agent
    if _agent is None:
        from .graph import TranscriptRAG
        _agent = TranscriptRAG(SETTINGS, corpus())
    return _agent


def retriever():
    global _retriever
    if _retriever is None:
        from .retriever import ChunkRetriever
        _retriever = ChunkRetriever(SETTINGS, corpus())
    return _retriever


def _save(upload: UploadFile, folder: str) -> Path:
    p = Path(folder) / Path(upload.filename or "upload.pdf").name
    with p.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return p


class Ask(BaseModel):
    question: str
    meeting_id: str | None = None


class Query(BaseModel):
    query: str
    meeting_id: str | None = None
    top_k: int = 8
    hops: int = 1
    expand: bool = True


@app.post("/ingest/transcript", summary="Upload a transcript PDF")
def ingest_transcript_ep(file: UploadFile = File(...), meeting_id: str | None = Form(None)):
    with tempfile.TemporaryDirectory() as d:
        return ingest_transcript(_save(file, d), SETTINGS, (meeting_id or "").strip() or None, corpus())


@app.post("/ingest/questionnaire", summary="Upload a questionnaire PDF (one chunk per question/answer row)")
def ingest_questionnaire_ep(file: UploadFile = File(...), meeting_id: str | None = Form(None)):
    with tempfile.TemporaryDirectory() as d:
        return ingest_questionnaire(_save(file, d), SETTINGS, (meeting_id or "").strip() or None, corpus())


@app.post("/ask", summary="Ask about already-ingested documents")
def ask(body: Ask):
    return agent().ask(body.question, body.meeting_id)


RUN_BODY = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["prompt"], "properties": {
        "prompt": {"type": "string", "description": "Your question/instructions (highest-priority source)"},
        "meeting_id": {"type": "string", "description": "Optional. Same id links transcript + questionnaire"},
        "transcript": {"type": "string", "format": "binary", "description": "Optional transcript PDF"},
        "questionnaire": {"type": "string", "format": "binary", "description": "Optional questionnaire PDF"},
    }}}}}}


@app.post("/run", openapi_extra=RUN_BODY,
          summary="One call: prompt + optional transcript PDF + optional questionnaire PDF")
async def run(request: Request):
    form = await request.form()
    prompt = str(form.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(422, "prompt is required")
    mid = str(form.get("meeting_id") or "").strip() or None
    uploads = {k: form.get(k) for k in ("transcript", "questionnaire")}
    uploads = {k: v for k, v in uploads.items() if v is not None and getattr(v, "filename", "")}

    with tempfile.TemporaryDirectory() as d:
        saved = {}
        for k, up in uploads.items():
            p = Path(d) / Path(up.filename).name
            p.write_bytes(await up.read())
            saved[k] = p
        if saved and not mid:
            mid = slug(next(iter(saved.values())).stem)       # both files share this id

        def work():
            ingested = []
            if "transcript" in saved:
                ingested.append(ingest_transcript(saved["transcript"], SETTINGS, mid, corpus()))
            if "questionnaire" in saved:
                ingested.append(ingest_questionnaire(saved["questionnaire"], SETTINGS, mid, corpus()))
            return ingested, agent().ask(prompt, mid)

        ingested, result = await run_in_threadpool(work)
    return {"meeting_id": mid, "ingested": ingested, **result}


@app.post("/search", summary="LLM-free retrieval (no answer generation)")
def search(q: Query):
    return retriever().search(q.query, q.meeting_id, q.top_k, q.hops, q.expand)


@app.get("/meetings", summary="What is stored (per meeting: transcript chunks / questionnaire rows)")
def meetings():
    return corpus().summary()
