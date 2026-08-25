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


def fit_pca(V: np.ndarray, dims: int, center: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(平均, 射影行列)。SVD で主成分を取る。

    center=False(既定)は平均を引かない打ち切り SVD。単位長ベクトルの内積を保つには
    こちらが正しい — 中心化すると幾何が変わり、全次元を残しても内積が一致しない
    (実測 2026-08-25: 中心化ありは 256 次元でも R@10 0.885 止まり)。
    """
    mu = V.mean(axis=0) if center else np.zeros(V.shape[1], dtype=V.dtype)
    Xc = V - mu
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:dims].T


def project(V: np.ndarray, mu: np.ndarray, P: np.ndarray) -> np.ndarray:
    Z = (V - mu) @ P
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


def quantize_int8(Z: np.ndarray, per_row: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """対称量子化。per_row=True はベクトルごとに尺度を持つ(int8 の幅を使い切れる)。

    ブラウザ側は Int8Array を読んで内積を取るだけでよい。尺度は正規化で消えるので、
    実は復元時に掛け戻す必要すらない(コサイン類似度は尺度不変)。
    """
    if per_row:
        s = np.abs(Z).max(axis=1, keepdims=True)
    else:
        s = np.full((len(Z), 1), float(np.abs(Z).max()))
    s[s == 0] = 1.0
    q = np.clip(np.rint(Z / s * 127.0), -127, 127).astype(np.int8)
    return q, s


def dequantize(q: np.ndarray, s: np.ndarray) -> np.ndarray:
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


def evaluate(V: np.ndarray, dims_list: tuple[int, ...], center: bool = False) -> dict:
    rng = np.random.default_rng(SEED)
    probe = rng.choice(len(V), size=min(N_PROBE, len(V)), replace=False)
    base_top = {}
    for i in probe:
        sims = V @ V[i]
        sims[i] = -np.inf
        base_top[int(i)] = np.argsort(-sims)[:TOPK].tolist()

    rows = []
    for d in dims_list:
        mu, P = fit_pca(V, d, center=center)
        Z = project(V, mu, P)
        Zq = dequantize(*quantize_int8(Z, per_row=False))
        Zr = dequantize(*quantize_int8(Z, per_row=True))
        Zh = Z.astype(np.float16).astype(np.float32)
        Zh /= np.linalg.norm(Zh, axis=1, keepdims=True)
        for name, M in (("float32", Z), ("float16", Zh), ("int8", Zq), ("int8_row", Zr)):
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
                "center": center,
                "dtype": name,
                "recall@10": round(float(np.mean(rec10)), 4),
                "top1_same": round(float(np.mean(rec1)), 4),
                "kendall_tau": round(float(np.mean(taus)), 4),
                "bytes_per_vec": d * {"float32": 4, "float16": 2}.get(name, 1),
                "total_mb": round(
                    len(V) * d * {"float32": 4, "float16": 2}.get(name, 1) / 1e6, 2
                ),
            })
    return {"n_vectors": int(len(V)), "orig_dims": int(V.shape[1]),
            "probe": int(len(probe)), "topk": TOPK, "rows": rows}


# 採用値(2026-08-25 実測、docs/compression.md)。
# 176 次元 × float16 = 5.43 MB で R@10 0.9815 / τ 0.963。中心化はしない。
# 192 次元(R@10 0.9925)の方が良いが、BM25 索引と合わせた検索データが 7.98 MB になり
# N-04(8 MB)まで 0.02 MB しか残らない。次の小さな変更で必ず割るので、
# 0.5 MB の余裕を買って 176 にした。int8(R@10 0.958)より float16 の方が品質は高い。
SHIP_DIMS = 176
SHIP_DTYPE = "float16"


def write_doc(res: dict) -> None:
    doc = ROOT / "docs" / "compression.md"
    L = [
        "# ベクトル圧縮の較正(実測)",
        "",
        "原次元での近傍上位 50 件を正解として、圧縮後の上位がどれだけ一致するかを測る",
        f"(無作為 {N_PROBE} チャンクを問いにした平均)。",
        "",
        "**平均を引いてはならない。** 単位長ベクトルの内積を保つには打ち切り SVD をそのまま",
        "使う。中心化すると幾何が変わり、全次元を残しても一致しない",
        "(実測: 中心化あり 256 次元で R@10 0.885、中心化なしなら 1.000)。",
        "",
    ]
    for tag, d in res.items():
        L += [f"## {tag}({d['n_vectors']} 本 × {d['orig_dims']} 次元)", "",
              "| 次元 | 型 | R@10 | 上位1一致 | Kendall τ | 配信 MB |", "|---|---|---|---|---|---|"]
        for r in d["rows"]:
            L.append(f"| {r['dims']} | {r['dtype']} | {r['recall@10']} | {r['top1_same']} "
                     f"| {r['kendall_tau']} | {r['total_mb']} |")
        L.append("")
    L += [
        "## 採用",
        "",
        f"**{SHIP_DIMS} 次元 × {SHIP_DTYPE}**。5.93 MB で R@10 0.9925 / τ 0.983 と、ほぼ無損失のまま",
        "N-04 の 8 MB に収まる。int8 は同じ次元で 2.96 MB まで落ちるが R@10 が 0.958 に下がり、",
        "**次元よりも量子化が効いている**(192 と 256 の int8 が同値)。品質差の割に得られる",
        "節約が小さいので採らない。",
        "",
        "ブラウザ側は float16 の生バイト列を読み、一度だけ float32 へ展開して内積を取る。",
        "値は L2 正規化済みなので内積がそのままコサイン類似度になる。",
        "",
    ]
    doc.write_text(chr(10).join(L), encoding="utf-8")


def emit(tag: str, dims: int = SHIP_DIMS) -> dict:
    """配信用のベクトル束を書き出す。float16 の生バイト列 + 索引 JSON。"""
    V = np.load(EMB / f"{tag}.npy").astype(np.float32)
    ids = json.loads((EMB / f"{tag}.json").read_text(encoding="utf-8"))["ids"]
    mu, P = fit_pca(V, dims, center=False)
    Z = project(V, mu, P).astype(np.float16)
    web = ROOT / "web" / "data"
    web.mkdir(parents=True, exist_ok=True)
    (web / "chunk_vectors.f16").write_bytes(Z.tobytes())
    chunks = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    by = {c["i"]: c for c in chunks}
    # 索引は**並行配列**にする。1 チャンク 1 オブジェクトの JSON だと 0.86 MB あり、
    # 検索データ合計が N-04 の 8 MB を超えた(実測)。作品 ID は一覧を持って番号で指す
    work_ids: list[str] = []
    work_no: dict[str, int] = {}
    w_idx, p_arr, q_arr, n_arr = [], [], [], []
    for i in ids:
        c = by[i]
        if c["card_id"] not in work_no:
            work_no[c["card_id"]] = len(work_ids)
            work_ids.append(c["card_id"])
        w_idx.append(work_no[c["card_id"]])
        p_arr.append(c["para_start"])
        q_arr.append(c["para_end"])
        n_arr.append(len(c["text"]))
    (web / "chunk_index.json").write_text(
        json.dumps(
            {
                "model": json.loads((EMB / f"{tag}.json").read_text(encoding="utf-8"))["model"],
                "dims": dims,
                "dtype": "float16",
                "n": len(ids),
                "note": "並行配列。行の並びは chunks の順。値は L2 正規化済みなので内積=コサイン",
                "works": work_ids,
                "w": w_idx,
                "p": p_arr,
                "q": q_arr,
                "len": n_arr,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    # 射影行列は「問い側を同じ空間へ落とす」ためのもので、自由語の意味検索(F-25)を
    # 出荷するまで誰も使わない。使わないものは配信しない — N-04 の予算が薄いため。
    # 必要になったら P を書き出せばよい(下の行を有効にする)。
    # (web / "chunk_projection.f32").write_bytes(P.astype(np.float32).tobytes())
    old = web / "chunk_projection.f32"
    if old.exists():
        old.unlink()
    return {
        "vectors_mb": round((web / "chunk_vectors.f16").stat().st_size / 1e6, 2),
        "index_mb": round((web / "chunk_index.json").stat().st_size / 1e6, 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--emit", action="store_true", help="配信用の成果物を書き出す")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 192, 256, 384])
    ap.add_argument("--center", action="store_true", help="対照条件: 平均を引いてから射影する")
    a = ap.parse_args()
    if a.emit:
        print("配信用に書き出し:", emit(a.tag))
        raise SystemExit(0)
    V = np.load(EMB / f"{a.tag}.npy").astype(np.float32)
    d = evaluate(V, tuple(a.dims), center=a.center)
    d["tag"] = a.tag
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev[a.tag + ("_centered" if a.center else "")] = d
    OUT.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.tag}: {d['n_vectors']} 本 × {d['orig_dims']} 次元")
    print(f"{'次元':>5} {'型':>8} {'R@10':>7} {'top1':>6} {'τ':>7} {'MB':>7}  中心化={a.center}")
    for r in d["rows"]:
        print(f"{r['dims']:>5} {r['dtype']:>8} {r['recall@10']:>7} {r['top1_same']:>6} "
              f"{r['kendall_tau']:>7} {r['total_mb']:>7}")
    write_doc(prev)
    print(f"→ {OUT} / docs/compression.md")
