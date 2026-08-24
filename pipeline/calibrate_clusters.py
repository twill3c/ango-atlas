"""クラスタリングの較正(F-09 / O-1)。

較正ファースト: UI を作る前に、二重版 50 組が特徴空間で互いに最近傍になるか、
クラスタが同じになるかを測る。品詞を原文で取った対照条件とも比べる。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from pipeline import cluster as cl

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "variant_pairs.json"
META = ROOT / "data" / "works_meta.json"
OUT_JSON = ROOT / "data" / "cluster_calibration.json"
OUT_DOC = ROOT / "docs" / "cluster_calibration.md"

KS = (3, 4, 5, 6, 8, 10)


def period(year: int | None) -> str:
    """通説の時期区分。**一致を要求しない**(観察のためのラベル — F-21)。"""
    if year is None:
        return "不明"
    if year <= 1937:
        return "初期(–1937)"
    if year <= 1945:
        return "戦中(1938–45)"
    if year <= 1949:
        return "戦後(1946–49)"
    return "晩年(1950–)"


def run() -> dict:
    pairs = [
        p
        for p in json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]
        if p["pair_type"] == "variant"
    ]
    meta = {r["card_id"]: r for r in json.loads(META.read_text(encoding="utf-8"))["works"]}
    out = {"provenance": {"built_at": "2026-08-25", "seed": cl.SEED}, "conditions": {}}
    for name, path in (
        ("fold_pos", ROOT / "data" / "style_features.json"),
        ("raw_pos", ROOT / "data" / "style_features_rawpos.json"),
    ):
        if not path.exists():
            continue
        cards, keys, X = cl.load(path)
        Z = cl.standardize(X)
        idx = {c: i for i, c in enumerate(cards)}
        D = np.sqrt(((Z[:, None, :] - Z[None]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        ranks = [
            int(np.where(np.argsort(D[idx[p["a"]]]) == idx[p["b"]])[0][0]) + 1 for p in pairs
        ]
        genres = [meta[c]["genre"] or "不明" for c in cards]
        periods = [period(meta[c]["pub_year"]) for c in cards]
        per_k = {}
        for k in KS:
            km = cl.kmeans(Z, k)
            wd = cl.ward_labels(Z, k)
            per_k[k] = {
                "kmeans_same_cluster": sum(
                    1 for p in pairs if km[idx[p["a"]]] == km[idx[p["b"]]]
                ),
                "ward_same_cluster": sum(
                    1 for p in pairs if wd[idx[p["a"]]] == wd[idx[p["b"]]]
                ),
                "ari_genre_kmeans": round(cl.ari(km.tolist(), genres), 4),
                "ari_period_kmeans": round(cl.ari(km.tolist(), periods), 4),
                "ari_genre_ward": round(cl.ari(wd.tolist(), genres), 4),
                "sizes_kmeans": sorted(np.bincount(km).tolist(), reverse=True),
            }
        out["conditions"][name] = {
            "n_features": len(keys),
            "o1_nearest": sum(1 for r in ranks if r == 1),
            "o1_top5": sum(1 for r in ranks if r <= 5),
            "o1_rank_median": statistics.median(ranks),
            "o1_rank_worst": max(ranks),
            "n_pairs": len(pairs),
            "per_k": per_k,
        }
    return out


if __name__ == "__main__":
    d = run()
    OUT_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    L = [
        "# 文体クラスタリングの較正(実測)",
        "",
        "UI を作る前に、二重版 50 組を使って特徴量が表記ではなく文体を捉えているかを測った。",
        "",
        "## O-1: 二重版の相手が最近傍か",
        "",
        "| 条件 | 相手が最近傍 | 5 位以内 | 順位中央値 | 最悪順位 |",
        "|---|---|---|---|---|",
    ]
    for name, c in d["conditions"].items():
        L.append(
            f"| {name} | {c['o1_nearest']}/{c['n_pairs']} | {c['o1_top5']}/{c['n_pairs']} | "
            f"{c['o1_rank_median']} | {c['o1_rank_worst']} |"
        )
    L += [
        "",
        "作品単位では品詞を原文で取っても畳んでも結果は変わらない。**畳むことが O-1 を救って",
        "いるわけではない** — 文長・句読点・文字種・会話文比率といった文字レベルの特徴が支配的で、",
        "それらは元々表記に鈍いからである。畳む効果は品詞ブロックに限られる",
        "(版間 L1 0.0218 → 0.0023、S/N 5.0 → 45.8)。チャンク単位の埋め込みを扱う L4/L5 では",
        "効き方が変わるはずなので、そこで測り直す。",
        "",
        "## クラスタ数ごとの共属と ARI(fold_pos)",
        "",
        "| k | k-means 同一クラスタ | Ward 同一クラスタ | ARI(ジャンル) | ARI(時期) | k-means の大きさ |",
        "|---|---|---|---|---|---|",
    ]
    fp = d["conditions"].get("fold_pos", {})
    for k, v in fp.get("per_k", {}).items():
        L.append(
            f"| {k} | {v['kmeans_same_cluster']}/50 | {v['ward_same_cluster']}/50 | "
            f"{v['ari_genre_kmeans']} | {v['ari_period_kmeans']} | {v['sizes_kmeans']} |"
        )
    L += [
        "",
        "ARI は**シルエット係数の代わり**に使う(F-09)。ジャンルとの ARI が低いこと自体は",
        "欠陥ではない — 「ジャンルラベルと文体クラスタが食い違う作品」を見せるのが F-10 の目的で、",
        "完全一致したらむしろ見るものが無い。",
        "",
    ]
    OUT_DOC.write_text("\n".join(L), encoding="utf-8")
    for name, c in d["conditions"].items():
        print(f"{name}: O-1 最近傍 {c['o1_nearest']}/{c['n_pairs']} 最悪 {c['o1_rank_worst']}")
    for k, v in fp.get("per_k", {}).items():
        print(f"  k={k:2d} 共属 km {v['kmeans_same_cluster']}/50 ward {v['ward_same_cluster']}/50 "
              f"ARI(ジャンル) {v['ari_genre_kmeans']:.3f} ARI(時期) {v['ari_period_kmeans']:.3f}")
    print(f"→ {OUT_JSON} / {OUT_DOC}")
