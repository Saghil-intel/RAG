# cli.py

import argparse
import json

from .config import SETTINGS


def _print_citation(c: dict) -> None:
    if c["source"] == "questionnaire":
        print(f"  [{c['tag']}] questionnaire row {c['row']}: {c['question'][:70]} -> {c['answer'][:70]}...")
    else:
        print(f"  [{c['tag']}] {c['ts']} {c['speaker']} ({c['org']}): {c['text'][:100]}...")


def main():
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="ingest a transcript PDF (optionally with its questionnaire)")
    i.add_argument("pdf")
    i.add_argument("--questionnaire", help="questionnaire PDF to ingest under the same meeting id")
    i.add_argument("--meeting-id")
    q = sub.add_parser("ingest-questionnaire")
    q.add_argument("pdf")
    q.add_argument("--meeting-id")
    s = sub.add_parser("search", help="LLM-free retrieval")
    s.add_argument("query")
    s.add_argument("--meeting")
    s.add_argument("-k", type=int, default=SETTINGS.rerank_k)
    s.add_argument("--hops", type=int, default=1, help="0 = single pass, 1 = one feedback hop")
    s.add_argument("--json", action="store_true")
    a = sub.add_parser("ask", help="LLM agent (Ollama) with multi-hop retrieval + citations")
    a.add_argument("question")
    a.add_argument("--meeting")
    a.add_argument("--trace", action="store_true")
    args = p.parse_args()

    if args.cmd in ("ingest", "ingest-questionnaire"):
        from .ingest import ingest_questionnaire, ingest_transcript, slug
        from pathlib import Path
        out = []
        if args.cmd == "ingest":
            mid = args.meeting_id or slug(Path(args.pdf).stem)
            out.append(ingest_transcript(args.pdf, SETTINGS, mid))
            if args.questionnaire:
                out.append(ingest_questionnaire(args.questionnaire, SETTINGS, mid))
        else:
            out.append(ingest_questionnaire(args.pdf, SETTINGS, args.meeting_id))
        print(json.dumps(out, indent=2))
    elif args.cmd == "ask":
        from .graph import TranscriptRAG
        r = TranscriptRAG(SETTINGS).ask(args.question, args.meeting)
        print(r["answer"], "\n")
        for c in r["citations"]:
            _print_citation(c)
        if args.trace:
            print("\n".join(["", *r["trace"]]))
    else:
        from .retriever import ChunkRetriever
        res = ChunkRetriever(SETTINGS).search(args.query, args.meeting, args.k, args.hops)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            for r in res:
                span = f"{r['start_ts']}-{r['end_ts']}" if r["source"] == "transcript" else f"row {r['row']}"
                print(f"\n=== [{r['source']}] {r['chunk_id']}  {span}  score={r['score']}  weighted={r['weighted_score']}")
                print(r["expanded_text"])


if __name__ == "__main__":
    main()
