"""文体の転換点検出(F-21)。

初出年ごとに文体特徴量の平均を取り、その系列に PELT(Pruned Exact Linear Time)を
かけて変化点を推定する。コストは二乗誤差(区間平均からの偏差)。

**通説の時期区分との一致は主張しない。** 帯として重ねて示し、重なったかどうかを
観察として記録するだけにする(SPEC F-21)。合致を要求するテストは書かない。

numpy だけで動く。乱数を使わないので再現性は自明(N-06)。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "data" / "style_features.json"
META = ROOT / "data" / "works_meta.json"
OUT = ROOT / "data" / "changepoints.json"

# 通説の時期区分(観察のために重ねる帯。一致を要求しない)
RECEIVED = [
    {"year": 1931, "label": "「風博士」で文壇の注目を浴びる"},
    {"year": 1938, "label": "「吹雪物語」前後の不遇期"},
    {"year": 1946, "label": "「堕落論」「白痴」で人気作家に"},
    {"year": 1950, "label": "巷談・捕物の連載期"},
]
NEAR = 1  # 何年までのずれを「重なった」とみなすか


# 罰則の倍率は**合成データで較正して決めた**(2026-08-25)。
# 純雑音 200 本での誤検出率 / 3σ の段差での検出力:
#   x1.0 → 79.0% / 98%   x1.5 → 22.5% / 100%   x2.0 → 4.0% / 100%
#   x2.5 →  1.5% / 98%   x3.0 →  0.0% / 100%
# 誤検出ゼロで検出力を落とさない x3.0 を採る。理論値(BIC)をそのまま使うと
# 雑音の 8 割で変化点を「発見」してしまう。
PENALTY_MULT = 3.0


def bic_penalty(n: int, dim: int) -> float:
    """BIC 相当の素の罰則(較正前)。"""
    return dim * math.log(max(n, 2))


def penalty_for(n: int, dim: int) -> float:
    """実際に使う罰則。較正した倍率を掛ける。"""
    return bic_penalty(n, dim) * PENALTY_MULT


def _cost(prefix: np.ndarray, prefix2: np.ndarray, a: int, b: int) -> float:
    """[a, b) の二乗誤差和。前計算した累積和から O(1) で出す。"""
    n = b - a
    if n <= 0:
        return 0.0
    s = prefix[b] - prefix[a]
    s2 = prefix2[b] - prefix2[a]
    return float(np.sum(s2 - (s * s) / n))


def pelt(x: np.ndarray, penalty: float) -> list[int]:
    """PELT。戻り値は変化点の位置(区間の開始位置、0 と n は含まない)。"""
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(X)
    prefix = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    prefix2 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X**2, axis=0)])

    F = [0.0] + [math.inf] * n
    cp: list[list[int]] = [[] for _ in range(n + 1)]
    candidates = [0]
    for t in range(1, n + 1):
        best, best_s = math.inf, 0
        for s in candidates:
            v = F[s] + _cost(prefix, prefix2, s, t) + penalty
            if v < best:
                best, best_s = v, s
        F[t] = best
        cp[t] = cp[best_s] + ([best_s] if best_s > 0 else [])
        # 枝刈り: 将来どの t' でも最適になりえない候補を落とす
        candidates = [
            s for s in candidates
            if F[s] + _cost(prefix, prefix2, s, t) <= F[t]
        ] + [t]
    return cp[n]


def series() -> tuple[list[int], np.ndarray, list[str]]:
    """初出年ごとの文体特徴量の平均(標準化済み)。"""
    feats = json.loads(FEATS.read_text(encoding="utf-8"))
    meta = {r["card_id"]: r for r in json.loads(META.read_text(encoding="utf-8"))["works"]}
    keys = [k for k in feats["keys"] if k not in ("n_chars", "n_chunks")]
    by_year: dict[int, list[list[float]]] = {}
    for cid, f in feats["features"].items():
        y = meta[cid]["pub_year"]
        if y is None:
            continue
        by_year.setdefault(y, []).append([f.get(k, 0.0) for k in keys])
    years = sorted(by_year)
    M = np.array([np.mean(by_year[y], axis=0) for y in years])
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    return years, (M - M.mean(axis=0)) / sd, keys


def build() -> dict:
    years, M, keys = series()
    # 特徴量が多いと罰則が効きすぎるので、主成分の上位だけを使う
    U, S, Vt = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
    Z = U[:, :4] * S[:4]
    cps = pelt(Z, penalty_for(len(years), Z.shape[1]))
    out = []
    for i in cps:
        y = years[i]
        near = [r for r in RECEIVED if abs(r["year"] - y) <= NEAR]
        out.append({
            "index": int(i),
            "year": int(y),
            "received_nearby": near,
        })
    return {
        "provenance": {
            "built_at": "2026-08-25",
            "method": "初出年ごとの文体特徴量平均 → 標準化 → 主成分 4 → PELT",
            "penalty_mult": PENALTY_MULT,
            "penalty_calibration": "純雑音 200 本で誤検出 0%、3σ の段差で検出力 100%(2026-08-25)",
            "note": "通説区分との一致は主張しない。重なりの有無を観察として記録するだけ",
        },
        "years": [int(y) for y in years],
        "n_features": len(keys),
        "features": keys,
        "pc_series": [[round(float(v), 4) for v in row] for row in Z],
        "changepoints": out,
        "received_periods": RECEIVED,
        "overlap": {
            "detected": len(out),
            "matched_received": sum(1 for c in out if c["received_nearby"]),
        },
    }


if __name__ == "__main__":
    d = build()
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"年 {d['years'][0]}–{d['years'][-1]}({len(d['years'])} 点)")
    for c in d["changepoints"]:
        near = "、".join(r["label"] for r in c["received_nearby"]) or "(通説区分の近傍になし)"
        print(f"  変化点 {c['year']}  {near}")
    print(f"検出 {d['overlap']['detected']} 件 / うち通説の近傍 {d['overlap']['matched_received']} 件 → {OUT}")
