"""Questionnaire PDF -> (question, answer) rows -> one chunk per row.

Docling detects the table; we read its cell grid directly (NOT export_to_dataframe, which may
swallow the first row as a header). Column 1 = question, column 2 = answer.
"""
from __future__ import annotations
import html
import re
from pathlib import Path

from .parsing import get_converter

# only a generic header row is dropped (never a real question)
HEADER_WORDS = {"question", "questions", "parameter", "parameters", "item", "field", "topic",
                "details", "description", "answer", "answers", "response", "responses", "s.no", "no", "no."}


def _clean(t: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(t or "")).strip()


def _table_to_grid(tbl) -> tuple[list[list[str]], set[int]]:
    d = tbl.data
    nr, nc = d.num_rows, d.num_cols
    grid = [[""] * nc for _ in range(nr)]
    merged: set[int] = set()
    for c in d.table_cells:
        r, k = c.start_row_offset_idx, c.start_col_offset_idx
        if r >= nr or k >= nc:
            continue
        text = _clean(c.text)
        if text:
            grid[r][k] = f"{grid[r][k]} {text}".strip()
        if nc > 1 and (getattr(c, "end_col_offset_idx", k + 1) - k) >= nc:
            merged.add(r)                                   # full-width cell = section heading
    return grid, merged


def _markdown_grid(md: str) -> list[list[str]]:
    grid = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [_clean(c) for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue                                        # |---|---| separator
        grid.append(cells)
    return grid


def _grid_to_pairs(grid, merged, pairs: list[list[str]], first_table: bool) -> None:
    for ri, row in enumerate(grid):
        cells = [_clean(c) for c in row]
        if not any(cells) or ri in merged:
            continue
        if first_table and ri == 0 and all(c.lower().strip(": ") in HEADER_WORDS for c in cells if c):
            continue
        q = cells[0]
        a = "; ".join(c for c in cells[1:] if c)
        if not q:                                           # continuation row after a page break
            if pairs and a:
                pairs[-1][1] = f"{pairs[-1][1]} {a}".strip()
            continue
        pairs.append([q, a or "(left blank)"])


def extract_qa_pairs(path: str | Path) -> list[tuple[str, str]]:
    doc = get_converter().convert(str(path)).document
    pairs: list[list[str]] = []
    for tbl in doc.tables:
        grid, merged = _table_to_grid(tbl)
        _grid_to_pairs(grid, merged, pairs, first_table=not pairs)
    if not pairs:                                           # fallback: pipe tables in the markdown
        _grid_to_pairs(_markdown_grid(doc.export_to_markdown()), set(), pairs, True)
    if not pairs:
        raise ValueError("No question/answer table found in the questionnaire PDF.")
    return [(q, a) for q, a in pairs]


def build_qa_chunks(pairs: list[tuple[str, str]], meeting_id: str, title: str) -> list[dict]:
    """One chunk per row = question + its answer (first and second column)."""
    chunks = []
    for i, (q, a) in enumerate(pairs):
        text = f"Question: {q}\nAnswer: {a}"
        chunks.append({
            "chunk_id": f"{meeting_id}:q{i}", "meeting_id": meeting_id, "source": "questionnaire",
            "row_index": i, "question": q, "answer": a, "text": text, "context": "",
            "embed_text": f"Source: questionnaire | Document: {title} | Row {i + 1}\n{text}",
            # compatibility with transcript-shaped fields
            "turn_ids": [], "turn_start": -1, "turn_end": -1, "start_ts": "", "end_ts": "",
            "speakers": [], "orgs": [], "topic_id": -1,
        })
    return chunks
