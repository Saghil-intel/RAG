"""LangGraph agent (Ollama) with bounded multi-hop retrieval over TWO sources.

Source priority:  1) the user's prompt   2) questionnaire (written answers)   3) transcript (spoken)

understand -> decompose -> retrieve -> evaluate
evaluate: insufficient & budget left -> retrieve again (follow-up queries built from evidence)
evaluate: sufficient or budget spent -> generate -> validate
validate: unfaithful -> generate again (bounded retries) ; else END

retrieve = per-source hybrid search + rerank (own quota per source), so neither source can crowd
out the other; priority is applied (a) as a soft weight when trimming evidence and (b) as a conflict
rule in the answer prompt.  Hard limits: max_iterations, max_retrieval_calls, max_revisions.
"""
from __future__ import annotations
import re
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Settings, SETTINGS
from .llm import check_ollama, get_llm, structured, text_of
from .store import Corpus, get_reranker


class Understanding(BaseModel):
    standalone_question: str = Field(description="Question rewritten to be self-contained")
    intent: Literal["lookup", "multi_hop", "summary"]


class SubQueries(BaseModel):
    queries: list[str] = Field(description="Independent search queries covering distinct facets")


class Sufficiency(BaseModel):
    sufficient: bool
    covered: str = Field(description="What the evidence already establishes")
    missing: list[str] = Field(description="Facts still needed")
    follow_up_queries: list[str] = Field(
        description="Targeted queries for the missing facts, using entities/numbers found in the evidence")


class Faithfulness(BaseModel):
    faithful: bool
    unsupported_claims: list[str]
    feedback: str


class RAGState(TypedDict, total=False):
    question: str
    meeting_id: Optional[str]
    standalone: str
    intent: str
    pending_queries: list[str]
    done_queries: list[str]
    evidence: dict            # key -> {source, meeting_id, idx, score, chunk_id}
    iteration: int
    retrieval_calls: int
    sufficient: bool
    missing: list[str]
    answer: str
    revisions: int
    feedback: str
    faithful: bool
    citations: list[dict]
    trace: list[str]


class TranscriptRAG:
    def __init__(self, cfg: Settings = SETTINGS, corpus: Corpus | None = None):
        self.cfg = cfg
        check_ollama(cfg)
        self.corpus = corpus or Corpus(cfg)
        self.reranker = get_reranker(cfg)
        self.llm = get_llm(cfg)
        self.graph = self._build()

    def _log(self, s: RAGState, msg: str) -> list[str]:
        return s.get("trace", []) + [msg]

    def _view(self, evidence: dict) -> tuple[str, dict]:
        """Evidence text: questionnaire rows first [Q1].., then transcript turns [E1]..,
        plus a tag -> source-item map used for citations."""
        cfg = self.cfg
        rows = [e for e in evidence.values() if e["source"] == "questionnaire"]
        turns = [e for e in evidence.values() if e["source"] != "questionnaire"]
        rows = sorted(rows, key=lambda e: e["score"], reverse=True)[: cfg.max_evidence_rows]
        rows.sort(key=lambda e: (e["meeting_id"], e["idx"]))
        turns = sorted(turns, key=lambda e: e["score"], reverse=True)[: cfg.max_evidence_turns]
        turns.sort(key=lambda e: (e["meeting_id"], e["idx"]))
        multi = len({e["meeting_id"] for e in rows + turns}) > 1
        alias: dict[str, dict] = {}
        parts: list[str] = []
        if rows:
            parts.append("== QUESTIONNAIRE (written answers; priority 2) ==")
            for n, e in enumerate(rows, 1):
                c = self.corpus.chunks[e["chunk_id"]]
                tag = f"Q{n}"
                alias[tag] = {"source": "questionnaire", "meeting_id": e["meeting_id"], "row": e["idx"] + 1,
                              "question": c["question"], "answer": c["answer"]}
                where = f"{e['meeting_id']} | " if multi else ""
                parts.append(f"[{tag} | {where}row {e['idx'] + 1}] Question: {c['question']} Answer: {c['answer']}")
        if turns:
            parts.append("== TRANSCRIPT (spoken conversation; priority 3) ==")
            prev = None
            for n, e in enumerate(turns, 1):
                t = self.corpus.turns[e["meeting_id"]][e["idx"]]
                if prev and (prev["meeting_id"] != e["meeting_id"] or prev["idx"] + 1 != e["idx"]):
                    parts.append("---")
                tag = f"E{n}"
                alias[tag] = {"source": "transcript", "meeting_id": e["meeting_id"],
                              **{k: t[k] for k in ("idx", "ts", "speaker", "org", "text")}}
                where = f"{e['meeting_id']} | " if multi else ""
                parts.append(f"[{tag} | {where}{t['ts']} | {t['speaker']} ({t['org']})] {t['text']}")
                prev = e
        return "\n".join(parts), alias

    # ------------------------------------------------------------ nodes
    def n_understand(self, s: RAGState) -> dict:
        u = structured(
            self.llm, Understanding,
            "You analyse questions about a customer engagement documented in a meeting transcript "
            "and a written questionnaire.\n"
            "Intent rules: 'lookup' = one fact (e.g. 'Which CPU was proposed?'); "
            "'multi_hop' = needs linked evidence from different parts (e.g. 'How does the proposal fix the "
            "current bottleneck?'); 'summary' = broad overview.\n"
            f"Question: {s['question']}\n"
            "Rewrite it as a standalone question and classify its intent.",
            Understanding(standalone_question=s["question"], intent="multi_hop"))
        return {"standalone": u.standalone_question or s["question"], "intent": u.intent, "iteration": 0,
                "retrieval_calls": 0, "revisions": 0, "evidence": {}, "done_queries": [],
                "trace": self._log(s, f"understand: intent={u.intent}")}

    def n_decompose(self, s: RAGState) -> dict:
        if s["intent"] == "lookup":
            qs = [s["standalone"]]
        else:
            r = structured(
                self.llm, SubQueries,
                f"Break this question into at most {self.cfg.max_subqueries} short, independent search "
                "queries, each targeting a distinct facet (e.g. the problem, the proposed solution, "
                "the numbers). Keep exact technical terms.\n"
                f"Question: {s['standalone']}",
                SubQueries(queries=[s["standalone"]]))
            qs = [q for q in r.queries if q.strip()][: self.cfg.max_subqueries] or [s["standalone"]]
        return {"pending_queries": qs, "trace": self._log(s, f"decompose: {qs}")}

    def n_retrieve(self, s: RAGState) -> dict:
        cfg = self.cfg
        budget = max(cfg.max_retrieval_calls - s.get("retrieval_calls", 0), 0)
        queries = s["pending_queries"][:budget]
        evidence = dict(s.get("evidence", {}))
        lookup = s.get("intent") == "lookup"
        plan = [("questionnaire", 2 if lookup else cfg.questionnaire_k, cfg.weight_questionnaire),
                ("transcript", 3 if lookup else cfg.rerank_k, cfg.weight_transcript)]
        found = {"questionnaire": 0, "transcript": 0}
        for q in queries:
            for source, k, w in plan:                        # own quota per source
                hits = self.corpus.hybrid_search(q, s.get("meeting_id"), source)
                if not hits:
                    continue
                ranked = self.reranker.rerank(q, [self.corpus.chunks[cid] for cid, _ in hits], k)
                for chunk, score in ranked:
                    score = float(score)
                    if source == "questionnaire":
                        if score < cfg.min_rerank_score:     # do not pad weak rows to fill the quota
                            continue
                        key = f"{chunk['meeting_id']}@q{chunk['row_index']}"
                        val = score * w
                        if key not in evidence or evidence[key]["score"] < val:
                            evidence[key] = {"source": source, "meeting_id": chunk["meeting_id"],
                                             "idx": chunk["row_index"], "chunk_id": chunk["chunk_id"], "score": val}
                        found[source] += 1
                    else:
                        for idx, core in self.corpus.expand(chunk, cfg.neighbor_window):
                            key = f"{chunk['meeting_id']}#{idx}"
                            val = (score if core else score * 0.5) * w
                            if key not in evidence or evidence[key]["score"] < val:
                                evidence[key] = {"source": source, "meeting_id": chunk["meeting_id"],
                                                 "idx": idx, "chunk_id": chunk["chunk_id"], "score": val}
                        found[source] += 1
        n_rows = sum(1 for e in evidence.values() if e["source"] == "questionnaire")
        return {"evidence": evidence, "done_queries": s.get("done_queries", []) + queries,
                "retrieval_calls": s.get("retrieval_calls", 0) + len(queries),
                "iteration": s.get("iteration", 0) + 1, "pending_queries": [],
                "trace": self._log(s, f"retrieve: {queries} -> {n_rows} questionnaire rows, "
                                      f"{len(evidence) - n_rows} transcript turns")}

    def n_evaluate(self, s: RAGState) -> dict:
        text, _ = self._view(s["evidence"])
        done = s.get("done_queries", [])
        r = structured(
            self.llm, Sufficiency,
            "Decide whether the evidence (questionnaire rows and transcript turns) is enough to fully "
            "answer the question.\n"
            "For cause/solution, comparison or before/after questions, BOTH sides must be covered, "
            "and each claimed link between them must be supported.\n"
            "If not sufficient, write follow-up queries that use specific entities, numbers or terms "
            f"DISCOVERED in the evidence. Do not repeat earlier queries: {done}\n\n"
            f"Question: {s['standalone']}\n\nEvidence:\n{text}",
            Sufficiency(sufficient=False, covered="", missing=[], follow_up_queries=[]))
        seen = {q.lower().strip() for q in done}
        new = []
        for q in r.follow_up_queries:
            if q.strip() and q.lower().strip() not in seen:
                new.append(q)
                seen.add(q.lower().strip())
        return {"sufficient": r.sufficient, "missing": r.missing,
                "pending_queries": new[: self.cfg.max_subqueries],
                "trace": self._log(s, f"evaluate: sufficient={r.sufficient} missing={r.missing} follow_ups={new}")}

    def route_after_eval(self, s: RAGState) -> str:
        cfg = self.cfg
        if (s["sufficient"] or s["iteration"] >= cfg.max_iterations
                or s["retrieval_calls"] >= cfg.max_retrieval_calls or not s["pending_queries"]):
            return "generate"
        return "retrieve"

    def n_generate(self, s: RAGState) -> dict:
        text, _ = self._view(s["evidence"])
        gaps = "" if s.get("sufficient") else (
            f"\nNOTE: retrieval budget ended with these gaps; state them plainly: {s.get('missing')}")
        fb = f"\nFix these problems from the previous draft: {s['feedback']}" if s.get("feedback") else ""
        r = self.llm.invoke(
            "Answer the user's request using ONLY the evidence below.\n"
            "SOURCE PRIORITY (highest first): 1) the user's own request, including any facts or "
            "assumptions the user states in it; 2) questionnaire answers [Q#]; 3) transcript turns [E#].\n"
            "- If sources disagree, follow the higher-priority source AND state the discrepancy "
            "briefly, citing both. If they agree or complement each other, cite both.\n"
            "- Answer exactly what was asked. Simple factual questions get 1-2 sentences. "
            "No unrequested background or recommendations.\n"
            "- Copy numbers, names and comparison wording exactly (e.g. 'at least', 'at or below').\n"
            "- Do NOT expand acronyms unless the evidence itself does.\n"
            "- Cite each claim with the tag that directly contains it, e.g. [Q2] or [E3].\n"
            "- If the evidence does not contain the answer, say 'Not stated in the provided documents.'\n"
            "- For questions linking a problem to a solution, state the link and cite both sides."
            f"{gaps}{fb}\n\nUser request: {s['standalone']}\n\nEvidence:\n{text}")
        return {"answer": text_of(r), "trace": self._log(s, "generate")}

    def n_validate(self, s: RAGState) -> dict:
        text, alias = self._view(s["evidence"])
        groups = re.findall(r"\[([^\]]*)\]", s["answer"])
        cited = sorted({t for g in groups for t in re.findall(r"\b[EQ]\d+\b", g)},
                       key=lambda x: (x[0] != "Q", int(x[1:])))
        bad = [t for t in cited if t not in alias]
        ok = Faithfulness(faithful=True, unsupported_claims=[], feedback="")
        r = ok if not self.cfg.llm_faithfulness_check else structured(
            self.llm, Faithfulness,
            "Check the answer against the evidence. List every claim that is not directly "
            "supported by the item it cites (or is uncited).\n\n"
            f"Request: {s['standalone']}\n\nAnswer:\n{s['answer']}\n\nEvidence:\n{text}", ok)
        faithful = r.faithful and not bad
        fb = "; ".join(r.unsupported_claims + [f"invalid citation {b}" for b in bad]) or r.feedback
        out = {"faithful": faithful, "feedback": "" if faithful else fb,
               "citations": [{"tag": t, **alias[t]} for t in cited if t in alias],
               "trace": self._log(s, f"validate: faithful={faithful}")}
        if not faithful:
            out["revisions"] = s.get("revisions", 0) + 1
        return out

    def route_after_validate(self, s: RAGState) -> str:
        return "generate" if (not s["faithful"] and s["revisions"] <= self.cfg.max_revisions) else END

    def _build(self):
        g = StateGraph(RAGState)
        for name, fn in [("understand", self.n_understand), ("decompose", self.n_decompose),
                         ("retrieve", self.n_retrieve), ("evaluate", self.n_evaluate),
                         ("generate", self.n_generate), ("validate", self.n_validate)]:
            g.add_node(name, fn)
        g.add_edge(START, "understand")
        g.add_edge("understand", "decompose")
        g.add_edge("decompose", "retrieve")
        g.add_edge("retrieve", "evaluate")
        g.add_conditional_edges("evaluate", self.route_after_eval, {"retrieve": "retrieve", "generate": "generate"})
        g.add_edge("generate", "validate")
        g.add_conditional_edges("validate", self.route_after_validate, {"generate": "generate", END: END})
        return g.compile()

    def ask(self, question: str, meeting_id: str | None = None) -> dict:
        out = self.graph.invoke({"question": question, "meeting_id": meeting_id},
                                config={"recursion_limit": 30})
        answer = out["answer"]
        if not out.get("faithful", True):
            answer += "\n\n(Warning: some statements could not be fully verified against the documents.)"
        return {"answer": answer, "citations": out.get("citations", []), "faithful": out.get("faithful"),
                "iterations": out.get("iteration"), "retrieval_calls": out.get("retrieval_calls"),
                "trace": out.get("trace", [])}
