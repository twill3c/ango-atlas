"""ベクトル圧縮(F-15)。

配信用の成果物が N-04(検索用データ 8 MB 以下)に収まり、
圧縮の損失が較正表どおりであることを確かめる。
"""
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline import compress as cp

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "data"


@pytest.mark.unit
def test_t415_projection_without_centering_is_lossless_at_full_rank():
    """T-415 / F-15: 中心化しない打ち切り SVD は、全次元を残せば内積を保つ。

    中心化すると保たない — それが 256 次元でも R@10 0.885 に留まった原因だった
    (実測 2026-08-25)。この性質をテストとして固定する。
    """
    rng = np.random.default_rng(0)
    V = rng.normal(size=(200, 32)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    mu, P = cp.fit_pca(V, 32, center=False)
    Z = cp.project(V, mu, P)
    assert np.allclose(Z @ Z.T, V @ V.T, atol=1e-5)
    # 中心化すると保たない
    mu2, P2 = cp.fit_pca(V, 32, center=True)
    Z2 = cp.project(V, mu2, P2)
    assert not np.allclose(Z2 @ Z2.T, V @ V.T, atol=1e-3)


@pytest.mark.unit
def test_t416_int8_roundtrip_preserves_direction():
    """T-416 / F-15: int8 量子化は向きをおおむね保つ(尺度は正規化で消える)。"""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(50, 64)).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    for per_row in (False, True):
        q, s = cp.quantize_int8(Z, per_row=per_row)
        back = cp.dequantize(q, s)
        cos = (back * Z).sum(axis=1)
        assert cos.min() > 0.99, f"per_row={per_row} で向きが崩れた: {cos.min()}"


@pytest.mark.validation
def test_t417_shipped_vectors_fit_the_budget():
    """T-417 / N-04・F-15: 配信用の検索データが 8 MB 以下。"""
    files = ["chunk_vectors.f16", "chunk_index.json", "chunk_projection.f32"]
    if not all((WEB / f).exists() for f in files):
        pytest.skip("配信用ベクトル未生成")
    total = sum((WEB / f).stat().st_size for f in files)
    assert total <= 8_000_000, f"検索データが {total/1e6:.2f} MB で N-04 を超える"


@pytest.mark.validation
def test_t418_shipped_vectors_match_the_index():
    """T-418 / F-15: 生バイト列の行数・次元が索引と一致し、正規化されている。"""
    if not (WEB / "chunk_vectors.f16").exists():
        pytest.skip("配信用ベクトル未生成")
    idx = json.loads((WEB / "chunk_index.json").read_text(encoding="utf-8"))
    raw = np.frombuffer((WEB / "chunk_vectors.f16").read_bytes(), dtype=np.float16)
    assert raw.size == idx["n"] * idx["dims"]
    V = raw.reshape(idx["n"], idx["dims"]).astype(np.float32)
    norms = np.linalg.norm(V, axis=1)
    assert np.allclose(norms, 1.0, atol=2e-3), f"正規化が崩れている: {norms.min()}–{norms.max()}"
    chunks = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    assert [c["i"] for c in idx["chunks"]] == [c["i"] for c in chunks]


@pytest.mark.validation
def test_t419_compression_choice_is_backed_by_measurement():
    """T-419 / F-15: 採用値が較正表の実測に裏付けられている。

    定数を SPEC に書くだけでなく、その行が測定結果として残っていることを確かめる。
    """
    p = ROOT / "data" / "compression_eval.json"
    if not p.exists():
        pytest.skip("compression_eval.json 未生成")
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = next(iter(d.values()))["rows"]
    ship = [r for r in rows if r["dims"] == cp.SHIP_DIMS and r["dtype"] == cp.SHIP_DTYPE]
    assert ship, f"採用値 {cp.SHIP_DIMS}/{cp.SHIP_DTYPE} の測定行が無い"
    r = ship[0]
    assert r["recall@10"] >= 0.98, f"採用値の R@10 が低い: {r['recall@10']}"
    assert r["total_mb"] <= 8.0
    # int8 は同じ次元で品質が落ちることも記録として残っていること
    i8 = [x for x in rows if x["dims"] == cp.SHIP_DIMS and x["dtype"] == "int8"]
    assert i8 and i8[0]["recall@10"] < r["recall@10"]
