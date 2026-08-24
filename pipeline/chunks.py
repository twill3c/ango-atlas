"""チャンク分割(F-13)。

段落境界を優先しつつ、下限に達するまで段落を束ね、上限で切る。
長すぎる段落は句点で切り、句点が無ければ上限で切る。

**取りこぼしを作らない**のが最上位の制約(T-402)。結合すると原本文に一致する。
埋め込み用の文脈(作品名・ジャンル・初出年)は本文に混ぜず、`with_context` で
埋め込み時に前置きとして組み立てる — 表示には本文そのものを使うため。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "works_meta.json"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "chunks.json"

MIN_CHARS = 300
MAX_CHARS = 500
# 均しの下限。これを下回るチャンクは隣へ併合する(実測: 素の分割では 50 字未満が 126 件出た)
FLOOR = 120
SENT_END = "。！？"


def _split_long(text: str, max_chars: int) -> list[str]:
    """上限を超える塊を句点で切る。句点が無ければ上限で切る。"""
    out: list[str] = []
    buf = ""
    cur = ""
    for ch in text:
        cur += ch
        if ch in SENT_END:
            if len(buf) + len(cur) > max_chars and buf:
                out.append(buf)
                buf = cur
            else:
                buf += cur
            cur = ""
            while len(buf) > max_chars:
                out.append(buf[:max_chars])
                buf = buf[max_chars:]
    buf += cur
    while len(buf) > max_chars:
        out.append(buf[:max_chars])
        buf = buf[max_chars:]
    if buf:
        out.append(buf)
    return out


def split(
    paras: list[str], min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS
) -> list[dict]:
    """段落の並びをチャンクに切る。戻り値は text / para_start / para_end を持つ。"""
    chunks: list[dict] = []
    buf = ""
    start = 0
    for i, para in enumerate(paras):
        if len(buf) + len(para) > max_chars:
            if buf:
                chunks.append({"text": buf, "para_start": start, "para_end": i - 1})
                buf = ""
            if len(para) > max_chars:
                pieces = _split_long(para, max_chars)
                for k, piece in enumerate(pieces[:-1]):
                    chunks.append({"text": piece, "para_start": i, "para_end": i})
                buf = pieces[-1]
                start = i
                if len(buf) >= min_chars:
                    chunks.append({"text": buf, "para_start": i, "para_end": i})
                    buf = ""
                    start = i + 1
                continue
            buf = para
            start = i
            continue
        if not buf:
            start = i
        buf += para
        if len(buf) >= min_chars:
            chunks.append({"text": buf, "para_start": start, "para_end": i})
            buf = ""
            start = i + 1
    if buf:
        chunks.append({"text": buf, "para_start": start, "para_end": len(paras) - 1})
    return chunks


def _split_even(text: str, max_chars: int) -> list[str]:
    """上限以下で、なるべく均等な大きさに割る。句点の近くで切る。

    末尾のかけらを直前へ戻すと上限を超えてしまうので、最初から等分を狙う。
    """
    n = -(-len(text) // max_chars)
    if n <= 1:
        return [text]
    target = -(-len(text) // n)
    out, buf = [], ""
    for ch in text:
        buf += ch
        if len(buf) >= target and (ch in SENT_END or len(buf) >= max_chars):
            out.append(buf)
            buf = ""
    if buf:
        if out and len(out[-1]) + len(buf) <= max_chars:
            out[-1] += buf
        else:
            out.append(buf)
    return out


def rebalance(
    chunks: list[dict], min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS,
    floor: int = FLOOR,
) -> list[dict]:
    """極小チャンクを隣へ併合し、大きくなりすぎたら句点で二分する。

    段落を束ねる途中で上限に当たると下限未満のかけらが切り出される。埋め込みの単位として
    使いものにならないので、後処理で均す(L3 のチャンク末尾と同じ型 — HC-022)。
    作品全体が floor 未満のときは 1 チャンクのまま残す。
    """
    out = [dict(c) for c in chunks]
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, c in enumerate(out):
            if len(c["text"]) >= floor:
                continue
            j = i - 1 if i > 0 else i + 1
            if not (0 <= j < len(out)):
                continue
            a, b = (out[j], c) if j < i else (c, out[j])
            merged = {
                "text": a["text"] + b["text"],
                "para_start": a["para_start"],
                "para_end": b["para_end"],
            }
            lo, hi = min(i, j), max(i, j)
            out[lo:hi + 1] = [merged]
            changed = True
            break
    # 併合で上限を超えたものは、かけらが出ないよう**均等に**割り直す
    final: list[dict] = []
    for c in out:
        if len(c["text"]) <= max_chars:
            final.append(c)
            continue
        for piece in _split_even(c["text"], max_chars):
            final.append(
                {"text": piece, "para_start": c["para_start"], "para_end": c["para_end"]}
            )
    return final


def with_context(chunk: dict, title: str, genre: str | None, year: int | None) -> str:
    """埋め込み用の文字列。文脈を前置きしてから本文を置く(contextual chunking)。

    元のチャンクは書き換えない。
    """
    head = title
    if genre:
        head += f"／{genre}"
    if year:
        head += f"／{year}年"
    return f"{head}\n{chunk['text']}"


def paragraphs(card_id: str) -> list[str]:
    from pipeline import aozora_parser as ap

    with open(RAW / f"{card_id}.txt", encoding="utf-8", newline="") as f:
        body = ap.parse(f.read()).body_text
    return [l.strip() for l in body.replace(chr(13), "").split(chr(10)) if l.strip()]


def build() -> dict:
    meta = json.loads(META.read_text(encoding="utf-8"))["works"]
    out = []
    n = 0
    for r in meta:
        cid = r["card_id"]
        for c in rebalance(split(paragraphs(cid))):
            out.append(
                {
                    "i": n,
                    "card_id": cid,
                    "para_start": c["para_start"],
                    "para_end": c["para_end"],
                    "text": c["text"],
                }
            )
            n += 1
    return {
        "provenance": {"built_at": "2026-08-25", "source": "data/raw + works_meta"},
        "min_chars": MIN_CHARS,
        "max_chars": MAX_CHARS,
        "chunks": out,
    }


if __name__ == "__main__":
    import statistics

    d = build()
    OUT.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    lens = [len(c["text"]) for c in d["chunks"]]
    print(f"{len(lens)} チャンク / 中央値 {statistics.median(lens):.0f} 字 "
          f"/ 最小 {min(lens)} / 最大 {max(lens)} → {OUT}")
