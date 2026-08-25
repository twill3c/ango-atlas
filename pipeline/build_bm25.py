"""BM25 索引の生成(F-17)と、二重版の索引重複排除(F-18)。

ブラウザで検索するため、形態素解析器を持ち込まずに済む形にする:
  - 索引語は**名詞の表層形**に限る。2 文字以上、または 1 文字でも漢字のもの(桜・雪・酒・碁)。
    数詞は除く。目的は固有名詞と具体名の検索で、埋め込みが弱い領域を補うこと。
    活用語を入れると問い側の活用形と噛み合わないので入れない
    (実測 2026-08-25: 1 字漢字を足すと postings は 508k → 672k、索引は 1.55 → 約 1.9 MB)
  - 問い側の分かち書きは、**配信した語彙に対する最長一致**でブラウザ側が行う

配信物:
  web/data/bm25_terms.txt      改行区切りの語彙(辞書順)
  web/data/bm25_offsets.bin    Uint32 × (語数+1)。postings の byte 位置
  web/data/bm25_postings.bin   語ごとに (varint 文書番号の差分, varint tf) の並び
  web/data/bm25_docs.bin       Uint16 × 文書数。文書長(索引語の総出現数)
  web/data/bm25_meta.json      文書数・平均文書長・除外した作品

F-18: 二重版のうち片方だけを検索対象にする。どちらを残すかは実測で決める
(新字新仮名を優先。同一表記の重複は本文の長い方を残す)。除外は**索引からではなく
結果から**行えるよう、チャンク索引に印を付ける — リーダーでは両版とも読めるべきだから。
"""
from __future__ import annotations

import collections
import json
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.json"
META = ROOT / "data" / "works_meta.json"
PAIRS = ROOT / "data" / "variant_pairs.json"
WEB = ROOT / "web" / "data"

MIN_LEN = 2
K1 = 1.2
B = 0.75
_KANJI1 = re.compile(r"^[一-鿿々]$")


def is_index_term(surface: str, pos2: str) -> bool:
    """索引語にするか。1 文字は漢字のみ、数詞は除く。"""
    if pos2 == "数詞":
        return False
    return len(surface) >= MIN_LEN or bool(_KANJI1.match(surface))


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def skipped_cards() -> dict[str, str]:
    """検索結果から外す作品と、その理由。"""
    meta = {r["card_id"]: r for r in json.loads(META.read_text(encoding="utf-8"))["works"]}
    out = {}
    for p in json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]:
        a, b = p["a"], p["b"]
        if p["pair_type"] == "variant":
            drop = a if meta[a]["kana_type"] == "新字旧仮名" else b
            keep = b if drop == a else a
            out[drop] = f"二重版のため除外(新字新仮名の card{keep} を残す)"
        else:
            drop = a if (meta[a]["chars"] or 0) < (meta[b]["chars"] or 0) else b
            keep = b if drop == a else a
            out[drop] = f"重複作品のため除外(本文の長い card{keep} を残す)"
    return out


def tokenize_corpus(chunks: list[dict]) -> list[collections.Counter]:
    import fugashi

    tagger = fugashi.Tagger()
    docs = []
    for c in chunks:
        t = collections.Counter()
        for w in tagger(c["text"]):
            if w.feature.pos1 == "名詞" and is_index_term(w.surface, w.feature.pos2):
                t[w.surface] += 1
        docs.append(t)
    return docs


def build() -> dict:
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))["chunks"]
    skip = skipped_cards()
    docs = tokenize_corpus(chunks)

    # 除外作品のチャンクは索引に入れない(結果に出ないので postings も持たない)
    active = [i for i, c in enumerate(chunks) if c["card_id"] not in skip]
    inv: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    doclen = [0] * len(chunks)
    for i in active:
        d = docs[i]
        doclen[i] = sum(d.values())
        for term, tf in d.items():
            inv[term].append((i, tf))

    terms = sorted(inv)
    postings = bytearray()
    offsets = [0]
    for t in terms:
        plist = sorted(inv[t])
        postings += varint(len(plist))
        prev = 0
        for doc, tf in plist:
            postings += varint(doc - prev)
            postings += varint(min(tf, 65535))
            prev = doc
        offsets.append(len(postings))

    WEB.mkdir(parents=True, exist_ok=True)
    # 改行は LF 固定。write_text だと Windows で CRLF になり、JS 側の split で
    # 各語に CR が残って全て不一致になる(二実装照合で露見した)
    (WEB / "bm25_terms.txt").write_bytes(chr(10).join(terms).encode("utf-8"))
    (WEB / "bm25_offsets.bin").write_bytes(
        struct.pack(f"<{len(offsets)}I", *offsets)
    )
    (WEB / "bm25_postings.bin").write_bytes(bytes(postings))
    (WEB / "bm25_docs.bin").write_bytes(
        struct.pack(f"<{len(doclen)}H", *[min(x, 65535) for x in doclen])
    )
    active_len = [doclen[i] for i in active]
    meta = {
        "n_docs": len(active),
        "n_chunks": len(chunks),
        "avgdl": round(sum(active_len) / max(len(active_len), 1), 3),
        "n_terms": len(terms),
        "k1": K1,
        "b": B,
        "min_len": MIN_LEN,
        "term_rule": "名詞。2 文字以上、または 1 文字の漢字。数詞は除く",
        "skipped": skip,
        "note": "文書番号は chunks の並び。除外作品のチャンクは postings を持たない",
    }
    (WEB / "bm25_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return meta


if __name__ == "__main__":
    m = build()
    sizes = {
        p.name: round(p.stat().st_size / 1e6, 2)
        for p in sorted(WEB.glob("bm25_*"))
    }
    print(f"索引語 {m['n_terms']} / 文書 {m['n_docs']}(除外 {len(m['skipped'])} 作品)"
          f" / 平均文書長 {m['avgdl']}")
    print("配信量(MB):", sizes, "計", round(sum(sizes.values()), 2))
