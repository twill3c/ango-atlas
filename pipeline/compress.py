"""ベクトルの圧縮と損失測定(F-15)。

配信量(N-04: 検索用データ 8 MB 以下)に収めるため、PCA で次元を落として int8 に量子化する。
**採用値は実測で決める** — 圧縮後の近傍上位が原次元とどれだけ一致するか(Recall@10 と
順位相関)を測り、その表を根拠に次元と量子化を選ぶ。

numpy だけで動く(.venv でなくてもよい)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "embeddings"
OUT = ROOT / "data" / "compression_eval.json"

SEED = 20260825
N_PROBE = 400  # 損失測定に使う問い(チャンク)の数
TOPK = 50


def fit_pca(V: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """(平均, 射影行列)。SVD で主成分を取る。"""
    mu = V.mean(axis=0)
    Xc = V - mu
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:dims].T


def project(V: np.ndarray, mu: np.ndarray, P: np.ndarray) -> np.ndarray:
    Z = (V - mu) @ P
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


def quantize_int8(Z: np.ndarray) -> tuple[np.ndarray, float]:
    """対称量子化。scale は全体で 1 つ(復元時に掛け戻す)。"""
    s = float(np.abs(Z).max())
    if s == 0:
        s = 1.0
    q = np.clip(np.rint(Z / s * 127.0), -127, 127).astype(np.int8)
    return q, s


def dequantize(q: np.ndarray, s: float) -> np.ndarray:
    Z = q.astype(np.float32) * (s / 127.0)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


def kendall_tau(a: list[int], b: list[int]) -> float:
    """順位相関(共通要素だけで測る素朴な実装)。"""
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return 0.0
    ra = {x: i for i, x in enumerate(a)}
    rb = {x: i for i, x in enumerate(b)}
    con = dis = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            s = (ra[x] - ra[y]) * (rb[x] - rb[y])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    return (con - dis) / (con + dis) if con + dis else 0.0


def evaluate(V: np.ndarray, dims_list: tuple[int, ...]) -> dict:
    rng = np.random.default_rng(SEED)
    probe = rng.choice(len(V), size=min(N_PROBE, len(V)), replace=False)
    base_top = {}
    for i in probe:
        sims = V @ V[i]
        sims[i] = -np.inf
        base_top[int(i)] = np.argsort(-sims)[:TOPK].tolist()

    rows = []
    for d in dims_list:
        mu, P = fit_pca(V, d)
        Z = project(V, mu, P)
        q, s = quantize_int8(Z)
        Zq = dequantize(q, s)
        for name, M in (("float32", Z), ("int8", Zq)):
            rec10, rec1, taus = [], [], []
            for i in probe:
                sims = M @ M[i]
                sims[i] = -np.inf
                top = np.argsort(-sims)[:TOPK].tolist()
                base = base_top[int(i)]
                rec10.append(len(set(top[:10]) & set(base[:10])) / 10)
                rec1.append(1.0 if top[0] == base[0] else 0.0)
                taus.append(kendall_tau(base, top))
            rows.append({
                "dims": d,
                "dtype": name,
                "recall@10": round(float(np.mean(rec10)), 4),
                "top1_same": round(float(np.mean(rec1)), 4),
                "kendall_tau": round(float(np.mean(taus)), 4),
                "bytes_per_vec": d * (4 if name == "float32" else 1),
                "total_mb": round(len(V) * d * (4 if name == "float32" else 1) / 1e6, 2),
            })
    return {"n_vectors": int(len(V)), "orig_dims": int(V.shape[1]),
            "probe": int(len(probe)), "topk": TOPK, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 192, 256, 384])
    a = ap.parse_args()
    V = np.load(EMB / f"{a.tag}.npy").astype(np.float32)
    d = evaluate(V, tuple(a.dims))
    d["tag"] = a.tag
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev[a.tag] = d
    OUT.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.tag}: {d['n_vectors']} 本 × {d['orig_dims']} 次元")
    print(f"{'次元':>5} {'型':>8} {'R@10':>7} {'top1':>6} {'τ':>7} {'MB':>7}")
    for r in d["rows"]:
        print(f"{r['dims']:>5} {r['dtype']:>8} {r['recall@10']:>7} {r['top1_same']:>6} "
              f"{r['kendall_tau']:>7} {r['total_mb']:>7}")
    print(f"→ {OUT}")
