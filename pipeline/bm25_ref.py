"""BM25 の参照実装(F-17)。

**配信した binary をそのまま読む**。ブラウザ側の JS 実装と同じ入力・同じ式で動かし、
両者の上位が一致することを検査する(二実装照合)。ここが基準側。
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "data"


class Bm25:
    def __init__(self, web: Path = WEB):
        self.meta = json.loads((web / "bm25_meta.json").read_text(encoding="utf-8"))
        self.terms = (web / "bm25_terms.txt").read_text(encoding="utf-8").split("\n")
        off = (web / "bm25_offsets.bin").read_bytes()
        self.offsets = struct.unpack(f"<{len(off)//4}I", off)
        self.postings = (web / "bm25_postings.bin").read_bytes()
        dl = (web / "bm25_docs.bin").read_bytes()
        self.doclen = struct.unpack(f"<{len(dl)//2}H", dl)
        self.index = {t: i for i, t in enumerate(self.terms)}
        self.maxlen = max(len(t) for t in self.terms)

    def read_postings(self, term: str) -> list[tuple[int, int]]:
        i = self.index.get(term)
        if i is None:
            return []
        pos = self.offsets[i]
        end = self.offsets[i + 1]
        buf = self.postings

        def rd() -> int:
            nonlocal pos
            n = 0
            shift = 0
            while True:
                b = buf[pos]
                pos += 1
                n |= (b & 0x7F) << shift
                if not (b & 0x80):
                    return n
                shift += 7

        df = rd()
        out = []
        doc = 0
        for _ in range(df):
            doc += rd()
            tf = rd()
            out.append((doc, tf))
        assert pos == end, "postings の読み出し位置がずれている"
        return out

    def tokenize(self, query: str) -> list[str]:
        """配信した語彙に対する最長一致。ブラウザ側と同じ規則。"""
        out = []
        i = 0
        while i < len(query):
            hit = None
            # 下限は 1。索引には 1 文字の漢字語(桜・雪・碁)が入っている
            for n in range(min(self.maxlen, len(query) - i), 0, -1):
                cand = query[i : i + n]
                if cand in self.index:
                    hit = cand
                    break
            if hit:
                out.append(hit)
                i += len(hit)
            else:
                i += 1
        return out

    def search(self, query: str, top: int = 10) -> list[tuple[int, float]]:
        n = self.meta["n_docs"]
        avgdl = self.meta["avgdl"]
        k1, b = self.meta["k1"], self.meta["b"]
        scores: dict[int, float] = {}
        for term in self.tokenize(query):
            plist = self.read_postings(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc, tf in plist:
                dl = self.doclen[doc] or 1
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                scores[doc] = scores.get(doc, 0.0) + idf * tf * (k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top]


if __name__ == "__main__":
    import sys

    bm = Bm25()
    chunks = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    meta = {r["card_id"]: r for r in
            json.loads((ROOT / "data" / "works_meta.json").read_text(encoding="utf-8"))["works"]}
    q = sys.argv[1] if len(sys.argv) > 1 else "呉清源"
    print("分かち書き:", bm.tokenize(q))
    for doc, sc in bm.search(q, 8):
        c = chunks[doc]
        print(f"  {sc:6.2f} {meta[c['card_id']]['title'][:22]:24s} {c['text'][:44]}")
