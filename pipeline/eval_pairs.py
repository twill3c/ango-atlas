"""二重版チャンク対応(L4 の O-1)。

同一作品の新字新仮名版と新字旧仮名版のチャンクを、**仮名遣いを畳んだ本文の重なり**で
対応づける。埋め込みが表記に引きずられていないかを、人手ゼロで数千件規模で測れる。

正解は埋め込みと無関係に作る(畳んだ文字 4-gram の Jaccard)ので、循環参照にならない。
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline import kana_fold as kf

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.json"
PAIRS = ROOT / "data" / "variant_pairs.json"
OUT = ROOT / "data" / "eval_variant_chunks.json"

MIN_SCORE = 0.35  # 対応とみなす下限(実測の分布から決める)


def grams(s: str, n: int = 4) -> set[str]:
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def build() -> dict:
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))["chunks"]
    by_card: dict[str, list[dict]] = {}
    for c in chunks:
        by_card.setdefault(c["card_id"], []).append(c)
    pairs = [
        p
        for p in json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]
        if p["pair_type"] == "variant"
    ]
    rows = []
    scores = []
    for p in pairs:
        old = p["a"] if p["kana_a"] == "新字旧仮名" else p["b"]
        new = p["b"] if old == p["a"] else p["a"]
        if old not in by_card or new not in by_card:
            continue
        gn = {c["i"]: grams(kf.fold(c["text"])) for c in by_card[new]}
        for c in by_card[old]:
            go = grams(kf.fold(c["text"]))
            best, bi = 0.0, None
            for j, g in gn.items():
                if not (go | g):
                    continue
                s = len(go & g) / len(go | g)
                if s > best:
                    best, bi = s, j
            scores.append(best)
            if bi is not None and best >= MIN_SCORE:
                rows.append(
                    {"query_chunk": c["i"], "gold_chunk": bi, "score": round(best, 4),
                     "old_card": old, "new_card": new}
                )
    scores.sort()
    return {
        "provenance": {
            "built_at": "2026-08-25",
            "method": "畳んだ本文の 4-gram Jaccard で最良対応を取る",
            "min_score": MIN_SCORE,
        },
        "n_candidates": len(scores),
        "score_quantiles": {
            "p10": round(scores[len(scores) // 10], 4),
            "p50": round(scores[len(scores) // 2], 4),
            "p90": round(scores[len(scores) * 9 // 10], 4),
        },
        "pairs": rows,
    }


if __name__ == "__main__":
    d = build()
    OUT.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"対応 {len(d['pairs'])} 件 / 候補 {d['n_candidates']} 件  分位 {d['score_quantiles']} → {OUT}")
