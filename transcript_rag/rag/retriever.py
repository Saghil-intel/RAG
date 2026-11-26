"""LLM-free retrieval: per-source hybrid search -> (optional feedback hop) -> rerank ->
neighbour expansion -> priority-weighted merge (questionnaire > transcript)."""
from __future__ import annotations
from collections import Counter

from .config import Settings, SETTINGS
from .store import Corpus, get_reranker, lex_tokens


class ChunkRetriever:
    def __init__(self, cfg: Settings = SETTINGS, corpus: Corpus | None = None):
        self.cfg = cfg
        self.corpus = corpus or Corpus(cfg)
        self.reranker = get_reranker(cfg)

    # -- pseudo-relevance feedback: distinctive terms from the best hits become a 2nd query
    def _feedback_terms(self, query: str, chunks: list[dict], n_terms: int = 6) -> list[str]:
        bm = self.corpus.bm25
        if bm is None:
            return []
        q = set(lex_tokens(query))
        tf: Counter = Counter()
        for c in chunks:
            tf.update(t for t in set(lex_tokens(c["text"])) if t not in q and len(t) > 2)
        scored = {t: n * bm.idf.get(t, 0.0) for t, n in tf.items() if bm.idf.get(t, 0.0) > 0}
        return [t for t, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)[:n_terms]]

    def _candidates(self, query: str, meeting_id, source) -> list[str]:
        return [cid for cid, _ in self.corpus.hybrid_search(query, meeting_id, source)]

    def search(self, query: str, meeting_id: str | None = None, top_k: int | None = None,
               hops: int = 1, expand: bool = True) -> list[dict]:
        """hops=0: single pass. hops=1: one bounded feedback pass. top_k applies to the transcript;
        the questionnaire has its own quota (cfg.questionnaire_k)."""
        cfg, corpus = self.cfg, self.corpus
        plan = [("questionnaire", cfg.questionnaire_k, cfg.weight_questionnaire),
                ("transcript", top_k or cfg.rerank_k, cfg.weight_transcript)]
        out = []
        for source, k, weight in plan:
            pool = self._candidates(query, meeting_id, source)
            if not pool:
                continue
            for _ in range(max(0, min(hops, 2))):
                first = self.reranker.rerank(query, [corpus.chunks[c] for c in pool], 3)
                terms = self._feedback_terms(query, [c for c, _ in first])
                if not terms:
                    break
                pool += [c for c in self._candidates(query + " " + " ".join(terms), meeting_id, source)
                         if c not in pool]
            ranked = self.reranker.rerank(query, [corpus.chunks[c] for c in dict.fromkeys(pool)], k)
            for chunk, score in ranked:
                if source == "questionnaire" and score < cfg.min_rerank_score:
                    continue
                item = {k_: chunk.get(k_) for k_ in ("chunk_id", "meeting_id", "source", "start_ts", "end_ts",
                                                      "speakers", "orgs", "turn_ids", "topic_id", "text")}
                if source == "questionnaire":
                    item.update(row=chunk["row_index"] + 1, question=chunk["question"], answer=chunk["answer"])
                item["score"] = round(float(score), 4)
                item["weighted_score"] = round(float(score) * weight, 4)
                if expand:
                    if source == "questionnaire":
                        item["expanded_text"] = chunk["text"]
                    else:
                        turns = corpus.turns[chunk["meeting_id"]]
                        lines = []
                        for idx, core in corpus.expand(chunk, cfg.neighbor_window):
                            t = turns[idx]
                            lines.append(f"{'>' if core else ' '} [{t['ts']}] {t['speaker']} ({t['org']}): {t['text']}")
                        item["expanded_text"] = "\n".join(lines)      # '>' marks turns inside the chunk
                out.append(item)
        out.sort(key=lambda r: r["weighted_score"], reverse=True)
        return out
