"""web/ 生成物の検査(F-10 / F-11 / F-12)。

HC-022 の規律に従い、**全件に値を作った直後**の健全性と、
部分(段落)の集約が全体(本文)と一致することを確かめる。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "data"


def _works():
    p = WEB / "works.json"
    if not p.exists():
        pytest.skip("works.json 未生成")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.validation
def test_t308_works_payload_covers_corpus():
    """T-308 / F-11: 一覧が全作品を覆い、必須フィールドを持つ。"""
    w = _works()
    meta = json.loads((ROOT / "data" / "works_meta.json").read_text(encoding="utf-8"))["works"]
    assert {r["id"] for r in w["works"]} == {m["card_id"] for m in meta}
    need = {"id", "t", "g", "y", "c", "k", "cl", "mm", "x", "y2", "nb", "pd"}
    for r in w["works"]:
        assert need <= set(r), f"{r['id']} に欠けたフィールド: {need - set(r)}"
        assert 0 <= r["cl"] < w["k"]
        assert len(r["nb"]) >= 2
        # 近傍の先頭は自分自身(距離 0)ではなく、実在する card_id である
        assert all(n in {x["id"] for x in w["works"]} for n in r["nb"])


@pytest.mark.validation
def test_t309_reader_text_preserves_body():
    """T-309 / F-12: 段落に分けた本文が、パーサーの本文と 1 字も違わない。

    リーダーは表示だけの層なので、ここで字が落ちたら静かに壊れる(HC-022)。
    """
    from pipeline import aozora_parser as ap

    raw = ROOT / "data" / "raw"
    files = sorted((WEB / "texts").glob("*.json"))
    if not files:
        pytest.skip("texts 未生成")
    bad = []
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        with open(raw / f"{d['id']}.txt", encoding="utf-8", newline="") as f:
            body = ap.parse(f.read()).body_text
        joined = "".join(x[1] for para in d["paras"] for x in para if x[0] in ("t", "r"))
        expect = body.replace(chr(13), "").replace(chr(10), "")
        if joined != expect:
            bad.append((d["id"], len(joined), len(expect)))
    assert not bad, f"本文が段落化で変化した: {bad[:5]}"


@pytest.mark.validation
def test_t310_mismatch_flag_matches_neighbors():
    """T-310 / F-10: 「食い違い」旗が近傍多数決の定義どおりに立っている。"""
    import collections

    w = _works()
    by = {r["id"]: r for r in w["works"]}
    for r in w["works"]:
        near = [n for n in r["nb"] if n != r["id"]][:6]
        g = collections.Counter(by[n]["g"] for n in near if by[n]["g"])
        top = g.most_common(1)[0] if g else None
        expect = bool(r["g"] and top and top[0] != r["g"] and top[1] >= 4)
        assert r["mm"] == expect, f"{r['id']} の食い違い旗が定義と合わない"


@pytest.mark.validation
def test_t311_pca_and_clusters_are_reproducible():
    """T-311 / N-06: PCA とクラスタが再実行で一致する(seed 固定)。"""
    import numpy as np

    from pipeline import cluster as cl

    cards, keys, X = cl.load()
    Z = cl.standardize(X)
    a = cl.kmeans(Z, 8)
    b = cl.kmeans(Z, 8)
    assert (a == b).all()
    P1, _ = cl.pca(Z, 2)
    P2, _ = cl.pca(Z, 2)
    assert np.allclose(P1, P2)


@pytest.mark.validation
def test_t312_pages_render_without_error():
    """T-312 / F-10..12: 各ページのスクリプトを実データで実行し、例外・空描画が無い。

    ブラウザを開けない環境の代替。最小の DOM スタブで走らせる(tests/smoke_pages.js)。
    node が無ければ skip する。
    """
    import shutil
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node が無い")
    r = subprocess.run(
        ["node", str(ROOT / "tests" / "smoke_pages.js")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    for page in ("index.html", "lens.html", "reader.html"):
        assert f"OK {page}" in r.stdout, r.stdout
