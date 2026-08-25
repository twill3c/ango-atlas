"""検索(F-16 / F-17 / F-18)。

配信物を**そのまま**読む二実装照合が要。ブラウザで動く JS(web/search.js)と、
Python の参照実装 / numpy を突き合わせる。片方だけでは、改行コードのような
配信側の事故に気づけない(実際に bm25_terms.txt の CRLF で全滅した)。
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "data"

QUERIES = ["呉清源", "牧野信一", "競輪の大穴", "桜の花と山賊", "雪と酒", "碁", "ふるさとの雪"]


def _node(args: list[str]) -> dict:
    if not shutil.which("node"):
        pytest.skip("node が無い")
    if not (WEB / "bm25_terms.txt").exists():
        pytest.skip("BM25 索引が未生成")
    r = subprocess.run(
        ["node", str(ROOT / "tests" / "bm25_node.js"), *args],
        capture_output=True, text=True, cwd=ROOT, encoding="utf-8",
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


@pytest.mark.validation
def test_t501_bm25_two_implementations_agree():
    """T-501 / F-17: JS と Python の BM25 が同じ上位・同じ点数を返す。"""
    from pipeline.bm25_ref import Bm25

    js = _node(QUERIES)
    bm = Bm25()
    skip = bm.meta["skipped"]
    chunks = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    for q in QUERIES:
        assert js[q]["terms"] == bm.tokenize(q), f"{q} の分かち書きが食い違う"
        # Python 側は除外・多様性制限を持たないので、同じ条件に揃えてから比べる
        raw = bm.search(q, 400)
        per, want = {}, []
        for doc, sc in raw:
            w = chunks[doc]["card_id"]
            if w in skip or per.get(w, 0) >= 2:
                continue
            per[w] = per.get(w, 0) + 1
            want.append([doc, round(sc, 4)])
            if len(want) >= 10:
                break
        got = js[q]["hits"]
        assert [d for d, _ in got] == [d for d, _ in want], f"{q} の順位が食い違う"
        for (d1, s1), (d2, s2) in zip(got, want):
            assert abs(s1 - s2) < 0.01, f"{q} の点数が食い違う: {s1} 対 {s2}"


@pytest.mark.validation
def test_t502_vector_search_two_implementations_agree():
    """T-502 / F-16: JS のベクトル近傍が numpy の計算と一致する。

    配信した float16 をそのまま読み、同じ内積・同じ絞り込みで比べる。
    """
    import numpy as np

    if not (WEB / "chunk_vectors.f16").exists():
        pytest.skip("配信ベクトルが未生成")
    idx = json.loads((WEB / "chunk_index.json").read_text(encoding="utf-8"))
    meta = json.loads((WEB / "bm25_meta.json").read_text(encoding="utf-8"))
    V = np.frombuffer((WEB / "chunk_vectors.f16").read_bytes(), dtype=np.float16)
    V = V.reshape(idx["n"], idx["dims"]).astype(np.float32)
    probes = [7282, 5468, 26, 14364]
    js = _node([f"#{p}" for p in probes])
    for p in probes:
        sims = V @ V[p]
        order = np.argsort(-sims)
        per, want = {}, []
        for j in order:
            j = int(j)
            if j == p:
                continue
            w = idx["works"][idx["w"][j]]
            if w in meta["skipped"] or per.get(w, 0) >= 2:
                continue
            per[w] = per.get(w, 0) + 1
            want.append(j)
            if len(want) >= 10:
                break
        got = [d for d, _ in js[f"#{p}"]["hits"]]
        assert got == want, f"チャンク {p} の近傍が食い違う\nJS  {got}\nnp  {want}"


@pytest.mark.validation
def test_t503_dedup_covers_exactly_one_side_of_each_pair():
    """T-503 / F-18: 二重版・重複のどちらか一方だけが除外されている。"""
    meta = json.loads((WEB / "bm25_meta.json").read_text(encoding="utf-8"))
    pairs = json.loads((ROOT / "data" / "variant_pairs.json").read_text(encoding="utf-8"))["pairs"]
    works = {r["card_id"]: r for r in
             json.loads((ROOT / "data" / "works_meta.json").read_text(encoding="utf-8"))["works"]}
    skip = meta["skipped"]
    for p in pairs:
        dropped = [c for c in (p["a"], p["b"]) if c in skip]
        assert len(dropped) == 1, f"{p['title_a']} で除外が {len(dropped)} 件"
        if p["pair_type"] == "variant":
            assert works[dropped[0]]["kana_type"] == "新字旧仮名", "新仮名の側を残すこと"
    # 除外された作品のチャンクは postings を持たない(文書長 0)
    import struct

    dl = (WEB / "bm25_docs.bin").read_bytes()
    doclen = struct.unpack(f"<{len(dl)//2}H", dl)
    chunks = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    for i, c in enumerate(chunks):
        if c["card_id"] in skip:
            assert doclen[i] == 0, f"除外作品 {c['card_id']} のチャンクに文書長がある"


@pytest.mark.validation
def test_t504_results_are_diversified():
    """T-504 / F-16: 同一作品からの結果が上限を超えない。"""
    js = _node(QUERIES + ["#7282"])
    import collections

    idx = json.loads((WEB / "chunk_index.json").read_text(encoding="utf-8"))
    for key, r in js.items():
        works = collections.Counter(idx["works"][idx["w"][d]] for d, _ in r["hits"])
        assert not works or max(works.values()) <= 2, f"{key} で同一作品が {works.most_common(1)}"


@pytest.mark.validation
def test_t505_nearest_is_fast_enough():
    """T-505 / N-05: 近傍計算が 300 ms 以下(実測環境を記録する)。

    実測 2026-08-25(node、8 コア CPU、15,430 本 × 176 次元): 1 問あたり 15.3 ms。
    ブラウザは node より遅いことがあるので、余裕を見て 300 ms を上限にしている。
    """
    js = _node(["--time"])
    ms = js["--time"]["ms_per_query"]
    assert ms <= 300, f"1 問 {ms:.1f} ms は N-05 の 300 ms を超える"
