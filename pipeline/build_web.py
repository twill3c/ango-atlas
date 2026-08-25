"""web/ 用データの生成(F-10 / F-11 / F-12)。

出力:
  web/data/corpus_summary.json  コーパス実測サマリ
  web/data/works.json           作品一覧 + 文体マップ座標 + クラスタ + 近傍
  web/data/texts/{card_id}.json リーダー用の本文(段落 × ルビ・注記の構造)
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from pipeline import aozora_parser as ap
from pipeline import calibrate_clusters as cc
from pipeline import cluster as cl

ROOT = Path(__file__).resolve().parents[1]
WORKS = ROOT / "data" / "aozora_works.json"
META = ROOT / "data" / "works_meta.json"
PAIRS = ROOT / "data" / "variant_pairs.json"
CAL = ROOT / "data" / "fold_calibration.json"
CCAL = ROOT / "data" / "cluster_calibration.json"
RAW = ROOT / "data" / "raw"
WEB = ROOT / "web" / "data"

K = 8  # 文体クラスタ数。ARI はk=4が最大だが、案内としては細かい方が使える
NEIGHBORS = 6


def summary() -> dict:
    doc = json.loads(WORKS.read_text(encoding="utf-8"))
    works = doc["works"]
    got = [w for w in works if not w.get("external_host")]
    out = {
        "provenance": doc["provenance"],
        "listed": len(works),
        "fetched": len(got),
        "chars_total": sum(w.get("chars", 0) for w in got),
        "chars_max": max((w.get("chars", 0) for w in got), default=0),
        "kana_type": dict(collections.Counter(w["kana_type"] for w in works).most_common()),
        "text_kind": dict(
            collections.Counter(w.get("text_kind") for w in got).most_common()
        ),
    }
    m = json.loads(META.read_text(encoding="utf-8"))["works"]
    years = [r["pub_year"] for r in m if r["pub_year"]]
    out["genre"] = dict(collections.Counter(r["genre"] or "不明" for r in m).most_common())
    out["year_range"] = [min(years), max(years)]
    out["year_unknown"] = sum(1 for r in m if r["pub_year"] is None)
    pr = json.loads(PAIRS.read_text(encoding="utf-8"))
    out["pairs"] = {
        "variant": sum(1 for p in pr["pairs"] if p["pair_type"] == "variant"),
        "duplicate": sum(1 for p in pr["pairs"] if p["pair_type"] == "duplicate"),
        "calibration": pr["provenance"]["calibration"],
    }
    c = json.loads(CAL.read_text(encoding="utf-8"))
    out["fold"] = {
        "raw_median": c["stages"]["raw"]["median"],
        "folded_median": c["stages"]["g4_choon_full_fold"]["median"],
    }
    if CCAL.exists():
        cc_ = json.loads(CCAL.read_text(encoding="utf-8"))["conditions"]["fold_pos"]
        out["o1"] = {
            "nearest": cc_["o1_nearest"],
            "n_pairs": cc_["n_pairs"],
            "worst_rank": cc_["o1_rank_worst"],
            "ari_genre": cc_["per_k"][str(K)]["ari_genre_kmeans"]
            if str(K) in cc_["per_k"]
            else cc_["per_k"][K]["ari_genre_kmeans"],
        }
    return out


def works_payload() -> dict:
    meta = {r["card_id"]: r for r in json.loads(META.read_text(encoding="utf-8"))["works"]}
    cards, keys, X = cl.load()
    Z = cl.standardize(X)
    P, ratio = cl.pca(Z, 2)
    labels = cl.kmeans(Z, K)
    D = np.sqrt(((Z[:, None, :] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)

    # クラスタごとの優勢ジャンル(見出し用)
    dom = {}
    for j in range(K):
        g = collections.Counter(
            meta[cards[i]]["genre"] or "不明" for i in range(len(cards)) if labels[i] == j
        )
        dom[j] = g.most_common(1)[0][0] if g else "不明"

    pairs = {}
    for p in json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]:
        pairs[p["a"]] = {"card_id": p["b"], "type": p["pair_type"]}
        pairs[p["b"]] = {"card_id": p["a"], "type": p["pair_type"]}

    near_of = {cid: [cards[j] for j in np.argsort(D[i])[:NEIGHBORS]] for i, cid in enumerate(cards)}
    rows = []
    for i, cid in enumerate(cards):
        r = meta[cid]
        near = near_of[cid]
        # ジャンルの食い違い: クラスタ全体の優勢ジャンルではなく**近傍の多数決**で見る。
        # 局所的で解釈しやすく、UI 上で近傍を並べれば読者がその場で確かめられる
        ng = collections.Counter(meta[n]["genre"] for n in near if meta[n]["genre"])
        top = ng.most_common(1)[0] if ng else None
        mismatch = bool(r["genre"] and top and top[0] != r["genre"] and top[1] >= 4)
        rows.append(
            {
                "id": cid,
                "t": r["title"],
                "g": r["genre"],
                "y": r["pub_year"],
                "c": r["chars"],
                "k": r["kana_type"],
                "s": r["series"],
                "own": r["own_work"],
                "cl": int(labels[i]),
                # 文体クラスタの優勢ジャンルと自分のジャンルが食い違う作品(F-10 の主役)
                "mm": mismatch,
                "nbg": top[0] if top else None,
                "x": round(float(P[i, 0]), 3),
                "y2": round(float(P[i, 1]), 3),
                "nb": near,
                "pair": pairs.get(cid),
                "pd": cc.period(r["pub_year"]),
            }
        )
    return {
        "k": K,
        "cluster_genre": {str(j): dom[j] for j in range(K)},
        "pca_ratio": [round(float(ratio[0]), 4), round(float(ratio[1]), 4)],
        "features": keys,
        "works": rows,
    }


def texts() -> int:
    out = WEB / "texts"
    out.mkdir(parents=True, exist_ok=True)
    meta = {r["card_id"]: r for r in json.loads(META.read_text(encoding="utf-8"))["works"]}
    # 段落 → チャンクの対応(リーダーから「この一節から探す」へ渡すため)
    ch_path = ROOT / "data" / "chunks.json"
    by_card_chunks: dict[str, list] = {}
    if ch_path.exists():
        for c in json.loads(ch_path.read_text(encoding="utf-8"))["chunks"]:
            by_card_chunks.setdefault(c["card_id"], []).append(
                [c["para_start"], c["para_end"], c["i"]]
            )
    n = 0
    for cid, r in meta.items():
        with open(RAW / f"{cid}.txt", encoding="utf-8", newline="") as f:
            doc = ap.parse(f.read())
        paras = []
        cur: list = []
        for node in doc.body:
            if isinstance(node, ap.Text):
                pieces = node.raw.split("\n")
                for k, piece in enumerate(pieces):
                    if k:
                        if any(x[1].strip() if x[0] == "t" else True for x in cur):
                            paras.append(cur)
                        cur = []
                    piece = piece.replace("\r", "")
                    if piece:
                        cur.append(["t", piece])
            elif isinstance(node, ap.Ruby):
                cur.append(["r", node.base_text, node.ruby])
            else:
                cur.append(["a", node.raw])
        if cur and any(x[1].strip() if x[0] == "t" else True for x in cur):
            paras.append(cur)
        (out / f"{cid}.json").write_text(
            json.dumps(
                {
                    "id": cid,
                    "title": doc.title,
                    "subtitle": doc.subtitle,
                    "author": doc.author,
                    "genre": r["genre"],
                    "year": r["pub_year"],
                    "kana": r["kana_type"],
                    "source": doc.footer.split("\n")[0].strip(),
                    "chunks": by_card_chunks.get(cid, []),
                    "paras": paras,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        n += 1
    return n


if __name__ == "__main__":
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "corpus_summary.json").write_text(
        json.dumps(summary(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    w = works_payload()
    (WEB / "works.json").write_text(json.dumps(w, ensure_ascii=False), encoding="utf-8")
    n = texts()
    size = sum(p.stat().st_size for p in (WEB / "texts").glob("*.json"))
    print(f"works {len(w['works'])} 件 / 本文 {n} 件({size/1e6:.1f} MB)")
    print(f"PCA 寄与率 {w['pca_ratio']} / クラスタ優勢ジャンル {w['cluster_genre']}")
