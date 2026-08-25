"""主題(トピック)の抽出(F-19 / F-20)。

**BERTopic(UMAP → HDBSCAN)は使わない。** 実測(2026-08-25)で、このコーパスの埋め込み
空間には密度の谷が無かった:
    PCA 32 次元 / min_cluster_size 15〜60 → ノイズ 71〜85%
    PCA 5〜16 次元                        → 巨大トピック 2 個に潰れる
単一作家の散文は連続体で、クラスタ構造を持たない。「どこにも属さない段落をノイズとして
弾ける」という F-19 の当初の根拠が成り立たないので、**全チャンクに主題を割り当てる
k-means + c-TF-IDF** に変えた(seed 固定で再現する — N-06)。

形式名詞(こと・もの・ところ)は c-TF-IDF でも落ちきらないので、**明示の除外語**で外す。
df による機械的な足切りにすると「女」「人間」「日本」のような安吾の主題語まで落ちるため、
語を一つずつ見て決める(実測の df 上位 30 を読んで作った)。
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np

from pipeline import cluster as cl
from pipeline import compress as cp
from pipeline.build_bm25 import is_index_term

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "embeddings" / "ruri_raw.npy"
CHUNKS = ROOT / "data" / "chunks.json"
META = ROOT / "data" / "works_meta.json"
OUT = ROOT / "data" / "topics.json"

PCA_DIMS = 32
K = 20  # 主題の数。較正(docs/topics_calibration.md)で決める
TOPN = 10

# 形式名詞・一般語。df 上位 30 を読んで選んだ(2026-08-25)。
# 「女」「男」「人間」「日本」「心」「自分」は安吾の主題語なので**残す**。
STOP = {
    "こと", "もの", "ところ", "とき", "時", "方", "ため", "中", "上", "わけ", "はず",
    "つもり", "場合", "通り", "ほか", "他", "うち", "点", "面", "何", "よう", "事",
    "物", "所", "為", "以上", "以下", "一つ", "二つ", "三つ", "今", "あと", "先",
}


def terms_of(chunks: list[dict]) -> list[collections.Counter]:
    import fugashi

    tagger = fugashi.Tagger()
    out = []
    for c in chunks:
        t = collections.Counter()
        for w in tagger(c["text"]):
            s = w.surface
            if w.feature.pos1 == "名詞" and is_index_term(s, w.feature.pos2) and s not in STOP:
                t[s] += 1
        out.append(t)
    return out


def ctfidf(labels: np.ndarray, docs: list[collections.Counter], k: int,
           topn: int = TOPN, min_total: int = 5) -> list[list[str]]:
    """class-based TF-IDF。主題ごとに語を束ね、全体での頻度で割り引く。"""
    per = [collections.Counter() for _ in range(k)]
    for lab, d in zip(labels, docs):
        per[int(lab)].update(d)
    total = collections.Counter()
    for p in per:
        total.update(p)
    A = sum(total.values()) / max(k, 1)
    out = []
    for p in per:
        n = sum(p.values()) or 1
        sc = {
            t: (f / n) * math.log(1 + A / total[t])
            for t, f in p.items()
            if total[t] >= min_total
        }
        out.append([t for t, _ in sorted(sc.items(), key=lambda x: -x[1])[:topn]])
    return out


def reduced() -> np.ndarray:
    V = np.load(EMB).astype(np.float32)
    mu, P = cp.fit_pca(V, PCA_DIMS, center=False)
    return cp.project(V, mu, P)


def build(k: int = K) -> dict:
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))["chunks"]
    meta = {r["card_id"]: r for r in json.loads(META.read_text(encoding="utf-8"))["works"]}
    Z = reduced()
    labels = cl.kmeans(Z, k)
    docs = terms_of(chunks)
    words = ctfidf(labels, docs, k)

    # 主題ごとの代表チャンク(重心に最も近い 3 件)— F-20 の名づけの材料
    reps = []
    for j in range(k):
        idx = np.where(labels == j)[0]
        c = Z[idx].mean(axis=0)
        near = idx[np.argsort(-(Z[idx] @ c))[:3]]
        reps.append([int(x) for x in near])

    # 作品ごとの主題分布(作品ベクトルは平均プーリングではなく分布で表す — SPEC §7)
    by_work: dict[str, list[int]] = {}
    for lab, c in zip(labels, chunks):
        by_work.setdefault(c["card_id"], [0] * k)[int(lab)] += 1

    years = collections.defaultdict(lambda: [0] * k)
    genres = collections.defaultdict(lambda: [0] * k)
    for lab, c in zip(labels, chunks):
        r = meta[c["card_id"]]
        if r["pub_year"]:
            years[r["pub_year"]][int(lab)] += 1
        genres[r["genre"] or "不明"][int(lab)] += 1

    return {
        "provenance": {
            "built_at": "2026-08-25",
            "method": f"ruri 埋め込み → PCA {PCA_DIMS} 次元 → k-means k={k} → c-TF-IDF",
            "seed": cl.SEED,
            "note": "HDBSCAN は密度の谷が無く不適だった(docs/topics_calibration.md)",
        },
        "k": k,
        "sizes": np.bincount(labels, minlength=k).tolist(),
        "words": words,
        "reps": reps,
        "labels": labels.tolist(),
        "by_work": by_work,
        "by_year": {str(y): v for y, v in sorted(years.items())},
        "by_genre": dict(genres),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()
    d = build(a.k)
    OUT.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    order = np.argsort(-np.array(d["sizes"]))
    for j in order:
        print(f"  {d['sizes'][j]:5d}  {' / '.join(d['words'][j][:8])}")
    print(f"k={d['k']} → {OUT}")
