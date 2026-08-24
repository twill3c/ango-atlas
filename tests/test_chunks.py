"""チャンク分割(F-13)。

最重要の不変量は「取りこぼしが無い」こと — 全チャンクを結合すると原本文に一致する。
件数ではなくこの不変量で書く(HC-016)。
"""
import pytest

from pipeline import chunks as ck


@pytest.mark.unit
def test_t401_paragraph_boundary_preferred():
    """T-401 / F-13: 段落境界を優先し、下限に達するまで段落を束ねる。"""
    paras = ["あ" * 120, "い" * 120, "う" * 120, "え" * 600]
    cs = ck.split(paras, min_chars=300, max_chars=500)
    # 先頭 3 段落(360 字)は 1 チャンクに束ねられる
    assert cs[0]["text"] == "あ" * 120 + "い" * 120 + "う" * 120
    assert cs[0]["para_start"] == 0 and cs[0]["para_end"] == 2
    # 上限を超える段落は分割される
    assert all(len(c["text"]) <= 500 for c in cs)


@pytest.mark.unit
def test_t402_no_text_is_lost():
    """T-402 / F-13: 結合すると原本文に一致する(取りこぼし・重複が無い)。"""
    paras = ["あ" * 37, "い" * 812, "う" * 5, "え" * 499, "お" * 501]
    cs = ck.split(paras, min_chars=300, max_chars=500)
    assert "".join(c["text"] for c in cs) == "".join(paras)


@pytest.mark.unit
def test_t403_long_paragraph_splits_at_sentence_end():
    """T-403 / F-13: 長い段落は句点で切る。句点が無ければ上限で切る。"""
    p = ("あ" * 200 + "。") * 4
    cs = ck.split([p], min_chars=300, max_chars=500)
    assert all(c["text"].endswith("。") for c in cs[:-1]), [c["text"][-5:] for c in cs]
    assert "".join(c["text"] for c in cs) == p
    # 句点の無い長文は上限で切れる
    cs2 = ck.split(["か" * 1300], min_chars=300, max_chars=500)
    assert [len(c["text"]) for c in cs2] == [500, 500, 300]


@pytest.mark.unit
def test_t404_context_prefix():
    """T-404 / F-13: 埋め込み用の文脈は本文に混ぜず、前置きとして付ける。

    表示は本文そのものを使うため、前置きは保存せず埋め込み時に組み立てる。
    """
    c = {"text": "本文である。", "card_id": "42620"}
    s = ck.with_context(c, title="堕落論", genre="随筆・評論", year=1946)
    assert s.startswith("堕落論")
    assert "随筆・評論" in s and "1946" in s
    assert s.endswith("本文である。")
    assert c["text"] == "本文である。", "元のチャンクを書き換えてはならない"


@pytest.mark.validation
def test_t405_corpus_chunks_cover_every_work():
    """T-405 / F-13: 全作品がチャンク化され、本文が保存されている。"""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    p = root / "data" / "chunks.json"
    if not p.exists():
        pytest.skip("chunks.json 未生成")
    d = json.loads(p.read_text(encoding="utf-8"))
    meta = json.loads((root / "data" / "works_meta.json").read_text(encoding="utf-8"))["works"]
    assert {c["card_id"] for c in d["chunks"]} == {m["card_id"] for m in meta}
    assert all(len(c["text"]) <= d["max_chars"] for c in d["chunks"])
    # 各作品でチャンクを結合すると、リーダー用本文の段落結合と一致する
    from collections import defaultdict

    by = defaultdict(list)
    for c in d["chunks"]:
        by[c["card_id"]].append(c)
    from pipeline import aozora_parser as ap

    for cid in list(by)[:20]:
        with open(root / "data" / "raw" / f"{cid}.txt", encoding="utf-8", newline="") as f:
            body = ap.parse(f.read()).body_text
        expect = "".join(
            l.strip() for l in body.replace(chr(13), "").split(chr(10)) if l.strip()
        )
        got = "".join(c["text"] for c in sorted(by[cid], key=lambda x: x["i"]))
        assert got == expect, f"{cid} でチャンクが本文と一致しない"


@pytest.mark.unit
def test_t406_rebalance_removes_tiny_chunks():
    """T-406 / F-13: 極小チャンクを隣へ併合する(HC-022 の分布検査で発覚した型)。

    素の分割では 50 字未満が 126 件・最小 1 字だった(実測 2026-08-25)。
    """
    paras = ["あ" * 480, "い" * 20, "う" * 400]
    cs = ck.rebalance(ck.split(paras))
    assert all(len(c["text"]) >= ck.FLOOR for c in cs), [len(c["text"]) for c in cs]
    assert all(len(c["text"]) <= ck.MAX_CHARS for c in cs)
    assert "".join(c["text"] for c in cs) == "".join(paras)
    # 作品全体が下限未満なら 1 チャンクのまま残す
    one = ck.rebalance(ck.split(["短い。"]))
    assert [c["text"] for c in one] == ["短い。"]


@pytest.mark.validation
def test_t407_no_tiny_chunks_in_corpus():
    """T-407 / F-13: コーパス全体で極小チャンクが残らない(作品全体が短い場合を除く)。"""
    import json
    from collections import defaultdict
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "chunks.json"
    if not p.exists():
        pytest.skip("chunks.json 未生成")
    d = json.loads(p.read_text(encoding="utf-8"))
    by = defaultdict(list)
    for c in d["chunks"]:
        by[c["card_id"]].append(c)
    # FLOOR は目標であって硬い下限ではない。硬い下限は「隣と併合すると上限を超える」場合のみ
    # 許すこと。実測 2026-08-25: 15,430 チャンク中この例外は 3 件(93/117/118 字)。
    bad = []
    for cid, cs in by.items():
        if len(cs) == 1:
            continue
        cs = sorted(cs, key=lambda x: x["i"])
        for k, c in enumerate(cs):
            if len(c["text"]) >= ck.FLOOR:
                continue
            nb = [len(cs[j]["text"]) for j in (k - 1, k + 1) if 0 <= j < len(cs)]
            if any(n + len(c["text"]) <= d["max_chars"] for n in nb):
                bad.append((cid, len(c["text"]), nb))
    assert not bad, f"併合できるのに残っている極小チャンク: {bad[:8]}"
