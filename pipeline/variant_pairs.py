"""二重版ペアと重複作品の検出(F-06)。

**題名では確定できない**(HC-012)。実測 2026-08-25 では 50 組の二重版のうち 12 組が
題名違い(「石の思ひ」⇔「石の思い」、「をみな」⇔「おみな」、「歴史と現実」⇔「歴史と事実」等)で、
題名一致だけを使うと取りこぼす。逆に同題名の大半は連作であり誤検出になる。

そこで**漢字だけを残した骨格**の 4-gram Jaccard で照合する。仮名遣いの差は骨格に出ないので、
新字新仮名版と新字旧仮名版が同一作品なら極めて高い値になる。実測の分布は完全に二峰で、
一致群の最小 0.8239 / 非一致群の最大 0.0112 と 2 桁の隔たりがあった。閾値はその谷に置く。
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

from pipeline import aozora_parser as ap

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "works_meta.json"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "variant_pairs.json"

KANJI_RE = re.compile(r"[一-鿿々]")
NGRAM = 4
# 実測の谷に置く。一致群 0.8239 以上 / 非一致群 0.0112 以下(2026-08-25、全 513 件)
THRESHOLD = 0.5
LEN_RATIO = (0.6, 1.7)


def kanji_signature(body_text: str) -> str:
    """本文から漢字だけを残す。仮名遣いの差に依存しない骨格。"""
    return "".join(KANJI_RE.findall(body_text))


def ngrams(s: str, n: int = NGRAM) -> set[str]:
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def signatures(cards: list[str]) -> dict[str, str]:
    out = {}
    for cid in cards:
        with open(RAW / f"{cid}.txt", encoding="utf-8", newline="") as f:
            out[cid] = kanji_signature(ap.parse(f.read()).body_text)
    return out


def find_pairs(meta: list[dict], sigs: dict[str, str], threshold: float = THRESHOLD) -> list[dict]:
    by = {r["card_id"]: r for r in meta}
    grams = {cid: ngrams(s) for cid, s in sigs.items()}
    pairs = []
    for a, b in combinations(sorted(sigs), 2):
        la, lb = len(sigs[a]), len(sigs[b])
        if not la or not lb:
            continue
        if not (LEN_RATIO[0] < lb / la < LEN_RATIO[1]):
            continue
        j = jaccard(grams[a], grams[b])
        if j < threshold:
            continue
        ka, kb = by[a]["kana_type"], by[b]["kana_type"]
        pairs.append(
            {
                "a": a,
                "b": b,
                "title_a": by[a]["title"],
                "title_b": by[b]["title"],
                "kana_a": ka,
                "kana_b": kb,
                "score": round(j, 4),
                "pair_type": "variant" if ka != kb else "duplicate",
                "same_title": by[a]["title"] == by[b]["title"],
                "evidence": f"漢字骨格 {NGRAM}-gram Jaccard={j:.4f}(閾値 {threshold})",
            }
        )
    return sorted(pairs, key=lambda p: -p["score"])


def gap_report(meta: list[dict], sigs: dict[str, str]) -> dict:
    """閾値較正の根拠。一致群の最小と非一致群の最大を測る。"""
    grams = {cid: ngrams(s) for cid, s in sigs.items()}
    hi, lo = [], []
    for a, b in combinations(sorted(sigs), 2):
        la, lb = len(sigs[a]), len(sigs[b])
        if not la or not lb or not (LEN_RATIO[0] < lb / la < LEN_RATIO[1]):
            continue
        j = jaccard(grams[a], grams[b])
        (hi if j >= THRESHOLD else lo).append(j)
    return {
        "matched_min": min(hi) if hi else None,
        "unmatched_max": max(lo) if lo else None,
        "matched": len(hi),
        "compared": len(hi) + len(lo),
    }


if __name__ == "__main__":
    import collections

    meta = json.loads(META.read_text(encoding="utf-8"))["works"]
    sigs = signatures([r["card_id"] for r in meta])
    pairs = find_pairs(meta, sigs)
    gap = gap_report(meta, sigs)
    OUT.write_text(
        json.dumps(
            {
                "provenance": {
                    "method": f"漢字骨格 {NGRAM}-gram Jaccard",
                    "threshold": THRESHOLD,
                    "calibration": gap,
                    "built_at": "2026-08-25",
                },
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    t = collections.Counter(p["pair_type"] for p in pairs)
    print("ペア:", t.most_common())
    print("題名違い:", sum(1 for p in pairs if not p["same_title"]))
    print(f"較正: 一致群最小 {gap['matched_min']:.4f} / 非一致群最大 {gap['unmatched_max']:.4f}"
          f"({gap['compared']} 組比較)")
    print(f"→ {OUT}")
