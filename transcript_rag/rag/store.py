"""Models + storage + hybrid retrieval (source-aware: transcript | questionnaire).

Three SEPARATE retrieval paths, fused with Reciprocal Rank Fusion:
  dense   : BGE-M3 dense vectors in ChromaDB (cosine)
  sparse  : BGE-M3 learned lexical weights in our own inverted index
  bm25    : BM25 with a tokenizer that keeps 6737P, c7i.16xlarge, p95, 91.5 intact
Every search can be restricted by meeting_id and/or source.
Then a cross-encoder reranker (bge-reranker-v2-m3).
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import Settings, SETTINGS, SOURCE_PRIORITY

# ---------------------------------------------------------------- models

class Embedder:
    def __init__(self, cfg: Settings):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(cfg.embed_model, use_fp16=cfg.use_fp16)

    def encode(self, texts: list[str], batch_size: int = 16):
        out = self.model.encode(texts, batch_size=batch_size, max_length=1024,
                                return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        dense /= np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
        sparse = [{str(k): float(v) for k, v in lw.items()} for lw in out["lexical_weights"]]
        return dense, sparse


class Reranker:
    def __init__(self, cfg: Settings):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(cfg.rerank_model, use_fp16=cfg.use_fp16)

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[tuple[dict, float]]:
        if not chunks:
            return []
        scores = self.model.compute_score([[query, c["embed_text"]] for c in chunks], normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]


@lru_cache(maxsize=1)
def get_embedder(cfg: Settings = SETTINGS) -> Embedder:
    return Embedder(cfg)


@lru_cache(maxsize=1)
def get_reranker(cfg: Settings = SETTINGS) -> Reranker:
    return Reranker(cfg)

# ---------------------------------------------------------------- corpus

_TOK = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")


def lex_tokens(text: str) -> list[str]:
    toks = _TOK.findall(text.lower())
    extra = [p for t in toks for p in re.split(r"[.\-_]", t) if p and p != t]
    return toks + extra


def rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, cid in enumerate(ranking):
            fused[cid] += 1.0 / (k + r + 1)
    return fused


# per source: (chunks file, sparse file). Transcript files keep their original names.
_FILES = {"transcript": ("chunks.json", "sparse.json"),
          "questionnaire": ("q_chunks.json", "q_sparse.json")}


class Corpus:
    def __init__(self, cfg: Settings = SETTINGS):
        import chromadb
        self.cfg = cfg
        (cfg.data_dir / "meetings").mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(cfg.data_dir / "chroma"))
        self.col = self.client.get_or_create_collection(
            cfg.collection, metadata={"hnsw:space": "cosine"})
        self.reload()
        self._migrate_source_metadata()

    # ---- persistence
    def _mdir(self, mid: str) -> Path:
        return self.cfg.data_dir / "meetings" / mid

    @staticmethod
    def _meta(c: dict) -> dict:
        src = c.get("source", "transcript")
        return {"meeting_id": c["meeting_id"], "source": src, "priority": SOURCE_PRIORITY[src],
                "row_index": c.get("row_index", -1),
                "turn_start": c.get("turn_start", -1), "turn_end": c.get("turn_end", -1),
                "start_ts": c.get("start_ts", ""), "topic_id": c.get("topic_id", -1),
                "speakers": "|".join(c.get("speakers", [])), "orgs": "|".join(c.get("orgs", []))}

    def add_source(self, mid: str, source: str, chunks: list[dict], dense: np.ndarray,
                   sparse: list[dict], turns: list[dict] | None = None) -> None:
        """Store one source of one meeting. Re-ingesting replaces ONLY that source."""
        d = self._mdir(mid)
        d.mkdir(parents=True, exist_ok=True)
        cf, sf = _FILES[source]
        if (d / cf).exists():                                 # idempotent re-ingest
            old = [c["chunk_id"] for c in json.loads((d / cf).read_text())]
            try:
                self.col.delete(ids=old)
            except Exception:
                pass
        if turns is not None:
            (d / "turns.json").write_text(json.dumps(turns))
        (d / cf).write_text(json.dumps(chunks))
        (d / sf).write_text(json.dumps(dict(zip([c["chunk_id"] for c in chunks], sparse))))
        self.col.upsert(ids=[c["chunk_id"] for c in chunks], embeddings=dense.tolist(),
                        documents=[c["embed_text"] for c in chunks],
                        metadatas=[self._meta(c) for c in chunks])
        self.reload()

    def add_meeting(self, mid, turns, chunks, dense, sparse) -> None:   # backward compatible
        self.add_source(mid, "transcript", chunks, dense, sparse, turns=turns)

    def reload(self) -> None:
        self.turns: dict[str, list[dict]] = {}
        self.chunks: dict[str, dict] = {}
        sparse: dict[str, dict] = {}
        for d in sorted((self.cfg.data_dir / "meetings").glob("*")):
            if not d.is_dir() or not any((d / f[0]).exists() for f in _FILES.values()):
                continue
            self.turns[d.name] = (json.loads((d / "turns.json").read_text())
                                  if (d / "turns.json").exists() else [])
            for source, (cf, sf) in _FILES.items():
                if not (d / cf).exists():
                    continue
                for c in json.loads((d / cf).read_text()):
                    c.setdefault("source", source)
                    self.chunks[c["chunk_id"]] = c
                if (d / sf).exists():
                    sparse.update(json.loads((d / sf).read_text()))
        self.ids = list(self.chunks)
        self.inv: dict[str, list[tuple[str, float]]] = defaultdict(list)   # sparse inverted index
        for cid, w in sparse.items():
            for tok, val in w.items():
                self.inv[tok].append((cid, val))
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([lex_tokens(self.chunks[c]["text"] + " " + " ".join(self.chunks[c].get("speakers", [])))
                               for c in self.ids]) if self.ids else None

    def _migrate_source_metadata(self) -> None:
        """Chunks ingested before the questionnaire module have no `source` in Chroma: add it."""
        try:
            ids = [cid for cid, c in self.chunks.items() if c["source"] == "transcript"]
            if not ids:
                return
            got = self.col.get(ids=ids, include=["metadatas"])
            fix = [(i, m) for i, m in zip(got["ids"], got["metadatas"]) if m is not None and "source" not in m]
            if fix:
                self.col.update(ids=[i for i, _ in fix],
                                metadatas=[{**m, "source": "transcript", "priority": SOURCE_PRIORITY["transcript"]}
                                           for _, m in fix])
        except Exception:
            pass

    def summary(self) -> dict:
        out: dict[str, dict] = {}
        for c in self.chunks.values():
            row = out.setdefault(c["meeting_id"], {"transcript_chunks": 0, "questionnaire_rows": 0})
            row["questionnaire_rows" if c["source"] == "questionnaire" else "transcript_chunks"] += 1
        return out

    # ---- three retrieval paths
    def _ok(self, cid: str, meeting_id: str | None, source: str | None) -> bool:
        c = self.chunks[cid]
        return (meeting_id is None or c["meeting_id"] == meeting_id) and (source is None or c["source"] == source)

    def dense_search(self, qvec: np.ndarray, k: int, meeting_id: str | None, source: str | None = None) -> list[str]:
        eligible = sum(1 for cid in self.ids if self._ok(cid, meeting_id, source))
        if eligible == 0 or self.col.count() == 0:
            return []
        conds = []
        if meeting_id:
            conds.append({"meeting_id": meeting_id})
        if source:
            conds.append({"source": source})
        kw = {}
        if len(conds) == 1:
            kw["where"] = conds[0]
        elif len(conds) > 1:
            kw["where"] = {"$and": conds}
        res = self.col.query(query_embeddings=[qvec.tolist()], n_results=min(k, eligible), **kw)
        return res["ids"][0]

    def sparse_search(self, qw: dict, k: int, meeting_id: str | None, source: str | None = None) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for tok, qv in qw.items():
            for cid, dv in self.inv.get(tok, ()):
                if self._ok(cid, meeting_id, source):
                    scores[cid] += qv * dv
        return [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

    def bm25_search(self, query: str, k: int, meeting_id: str | None, source: str | None = None) -> list[str]:
        if not self.bm25:
            return []
        sc = self.bm25.get_scores(lex_tokens(query))
        order = np.argsort(-sc)
        return [self.ids[i] for i in order if sc[i] > 0 and self._ok(self.ids[i], meeting_id, source)][:k]

    def hybrid_search(self, query: str, meeting_id: str | None = None,
                      source: str | None = None) -> list[tuple[str, float]]:
        cfg = self.cfg
        dense, sparse = get_embedder(cfg).encode([query])
        fused = rrf([self.dense_search(dense[0], cfg.dense_k, meeting_id, source),
                     self.sparse_search(sparse[0], cfg.sparse_k, meeting_id, source),
                     self.bm25_search(query, cfg.bm25_k, meeting_id, source)], cfg.rrf_k)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:cfg.fused_k]

    # ---- context expansion (turn level; questionnaire rows are self-contained)
    def expand(self, chunk: dict, window: int) -> list[tuple[int, bool]]:
        if chunk.get("source", "transcript") != "transcript":
            return []
        n = len(self.turns[chunk["meeting_id"]])
        lo, hi = max(0, chunk["turn_start"] - window), min(n - 1, chunk["turn_end"] + window)
        core = set(chunk["turn_ids"])
        return [(i, i in core) for i in range(lo, hi + 1)]
