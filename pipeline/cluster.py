"""文体クラスタリング(F-09)。numpy だけで実装する(scipy/sklearn 非依存)。

- 標準化(z 得点)→ PCA(SVD)→ k-means(k-means++ 初期化・seed 固定)/ Ward 法
- ラベル(ジャンル・時期)との一致は**シルエット係数ではなく ARI** で見る(F-09)
- 乱数を使う手順は seed 固定で、再実行で同一出力を返す(N-06)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "data" / "style_features.json"
META = ROOT / "data" / "works_meta.json"

# 作品の大きさは文体ではないので特徴量に入れない
EXCLUDE = ("n_chars", "n_chunks")
SEED = 20260825


def load(path: Path = FEATS) -> tuple[list[str], list[str], np.ndarray]:
    d = json.loads(path.read_text(encoding="utf-8"))
    keys = [k for k in d["keys"] if k not in EXCLUDE]
    cards = sorted(d["features"])
    X = np.array([[d["features"][c].get(k, 0.0) for k in keys] for c in cards], dtype=float)
    return cards, keys, X


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def pca(X: np.ndarray, n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(主成分得点, 寄与率)。既定は PCA(UMAP は乱数依存のため既定にしない — N-06)。"""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2 / (len(X) - 1)
    return U[:, :n] * S[:n], var / var.sum()


def kmeans(X: np.ndarray, k: int, seed: int = SEED, iters: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # k-means++ 初期化
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(len(X), 1 / len(X))
        centers.append(X[rng.choice(len(X), p=probs)])
    C = np.array(centers)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(axis=1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    return labels


def ward(X: np.ndarray) -> list[tuple[int, int, float, int]]:
    """Ward 法の凝集型クラスタリング。戻り値は (a, b, 距離, 併合後の大きさ) の列。

    Lance-Williams 更新で O(n^2) の距離行列を保つ素朴な実装。n=513 で十分速い。
    """
    n = len(X)
    d = ((X[:, None, :] - X[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    size = np.ones(n)
    active = list(range(n))
    idx = {i: i for i in range(n)}
    merges = []
    nxt = n
    D = d.astype(float)
    for _ in range(n - 1):
        sub = D[np.ix_(active, active)]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = active[i], active[j]
        dist = float(sub[i, j])
        merges.append((idx[a], idx[b], dist**0.5, int(size[a] + size[b])))
        # Lance-Williams(Ward)
        for c in active:
            if c in (a, b):
                continue
            na, nb, nc = size[a], size[b], size[c]
            t = na + nb + nc
            D[a, c] = D[c, a] = (
                (na + nc) * D[a, c] + (nb + nc) * D[b, c] - nc * dist
            ) / t
        size[a] += size[b]
        idx[a] = nxt
        nxt += 1
        active.remove(b)
        D[b, :] = np.inf
        D[:, b] = np.inf
    return merges


def ward_labels(X: np.ndarray, k: int) -> np.ndarray:
    """Ward のデンドログラムを k 個に切る。"""
    n = len(X)
    merges = ward(X)
    parent = {}
    members = {i: [i] for i in range(n)}
    nxt = n
    for a, b, _dist, _sz in merges[: n - k]:
        members[nxt] = members.pop(a) + members.pop(b)
        nxt += 1
    labels = np.zeros(n, dtype=int)
    for lab, (_key, mem) in enumerate(sorted(members.items())):
        for m in mem:
            labels[m] = lab
    return labels


def ari(a, b) -> float:
    """調整ランド指数。ラベルは任意のハッシュ可能値でよい。"""
    a = list(a)
    b = list(b)
    ua = {v: i for i, v in enumerate(sorted(set(a), key=str))}
    ub = {v: i for i, v in enumerate(sorted(set(b), key=str))}
    m = np.zeros((len(ua), len(ub)))
    for x, y in zip(a, b):
        m[ua[x], ub[y]] += 1
    def c2(x):
        return x * (x - 1) / 2
    sij = c2(m).sum()
    si = c2(m.sum(axis=1)).sum()
    sj = c2(m.sum(axis=0)).sum()
    n = c2(len(a))
    exp = si * sj / n
    mx = (si + sj) / 2
    return float((sij - exp) / (mx - exp)) if mx != exp else 0.0
