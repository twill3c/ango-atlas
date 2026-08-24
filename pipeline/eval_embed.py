"""埋め込みの評価(F-14)。.venv で実行する。

2 つのオラクルで測る:
  A) 二重版チャンク対応(1,026 組・人手ゼロ)
     旧仮名版のチャンクを問いにして、対応する新仮名版のチャンクが取れるか。
     **表記への頑健性**を測る。正解は畳んだ本文の重なりで作ってあり、埋め込みとは無関係。
  B) 人手評価セット(35 問)
     チャンクを読んで内容を別の言葉で言い換えた問い。**意味検索の質**を測る。

どちらも Recall@1 / Recall@10 / MRR を出す。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "embeddings"
CHUNKS = ROOT / "data" / "chunks.json"
QUERIES = ROOT / "data" / "eval_queries.json"
VARIANTS = ROOT / "data" / "eval_variant_chunks.json"
OUT = ROOT / "data" / "embed_eval.json"


def load(tag: str) -> tuple[np.ndarray, list[int]]:
    V = np.load(EMB / f"{tag}.npy")
    ids = json.loads((EMB / f"{tag}.json").read_text(encoding="utf-8"))["ids"]
    return V, ids


def ranks_of(sims: np.ndarray, gold_rows: set[int], drop: set[int]) -> int | None:
    order = np.argsort(-sims)
    r = 0
    for j in order:
        if int(j) in drop:
            continue
        r += 1
        if int(j) in gold_rows:
            return r
        if r > 100:
            return None
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    got = [r for r in ranks if r is not None]
    return {
        "n": n,
        "recall@1": round(sum(1 for r in got if r == 1) / n, 4),
        "recall@5": round(sum(1 for r in got if r <= 5) / n, 4),
        "recall@10": round(sum(1 for r in got if r <= 10) / n, 4),
        "mrr": round(sum(1 / r for r in got) / n, 4),
        "median_rank": int(np.median(got)) if got else None,
    }


def eval_variants(V: np.ndarray, row_of: dict[int, int]) -> dict:
    ev = json.loads(VARIANTS.read_text(encoding="utf-8"))["pairs"]
    chunks = {c["i"]: c for c in json.loads(CHUNKS.read_text(encoding="utf-8"))["chunks"]}
    ranks = []
    for e in ev:
        qi, gi = e["query_chunk"], e["gold_chunk"]
        if qi not in row_of or gi not in row_of:
            continue
        # 同じ作品(=問い側の版)のチャンクは除いて探す
        same = {row_of[c] for c, v in chunks.items()
                if v["card_id"] == e["old_card"] and c in row_of}
        sims = V @ V[row_of[qi]]
        ranks.append(ranks_of(sims, {row_of[gi]}, same))
    return metrics(ranks)


def eval_queries(V: np.ndarray, row_of: dict[int, int], model: str) -> dict:
    from pipeline.embed import Encoder

    qs = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    ev = json.loads(VARIANTS.read_text(encoding="utf-8"))["pairs"]
    twin: dict[int, set[int]] = {}
    for e in ev:
        twin.setdefault(e["gold_chunk"], set()).add(e["query_chunk"])
        twin.setdefault(e["query_chunk"], set()).add(e["gold_chunk"])
    enc = Encoder(model)
    Q = enc.encode([e["q"] for e in qs], kind="query")
    ranks, detail = [], []
    for e, q in zip(qs, Q):
        gold = set(e["gold"])
        for g in list(gold):
            gold |= twin.get(g, set())  # 二重版の対応チャンクも正解に数える
        rows = {row_of[g] for g in gold if g in row_of}
        r = ranks_of(V @ q, rows, set())
        ranks.append(r)
        detail.append({"q": e["q"], "work": e["work"], "rank": r})
    m = metrics(ranks)
    m["detail"] = detail
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="例: ruri_raw e5_raw")
    a = ap.parse_args()
    out = {}
    for tag in a.tags:
        model = tag.split("_")[0]
        V, ids = load(tag)
        row_of = {c: i for i, c in enumerate(ids)}
        va = eval_variants(V, row_of)
        qa = eval_queries(V, row_of, model)
        out[tag] = {"variant_chunks": va, "queries": {k: v for k, v in qa.items() if k != "detail"},
                    "query_detail": qa["detail"]}
        print(f"== {tag}")
        print(f"   A 二重版 {va['n']} 件: R@1 {va['recall@1']} R@10 {va['recall@10']} MRR {va['mrr']}")
        print(f"   B 評価問 {qa['n']} 問: R@1 {qa['recall@1']} R@10 {qa['recall@10']} MRR {qa['mrr']}")
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev.update(out)
    OUT.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT}")
