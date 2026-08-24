"""二重版ペアの検出(F-06)。O-1 オラクルの素。

期待値は抽出出力から貼る(HC-019)。件数ではなく集合・不変量で書く(HC-016)。
"""
import json
from pathlib import Path

import pytest

from pipeline import variant_pairs as vp

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "variant_pairs.json"


def _pairs():
    if not PAIRS.exists():
        pytest.skip("variant_pairs.json 未生成")
    return json.loads(PAIRS.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_t207_kanji_signature_drops_kana():
    """T-207 / F-06: 骨格は漢字だけを残す。仮名遣いの差が消えることが要点。"""
    old = "私は海をだきしめてゐたい。"
    new = "私は海をだきしめていたい。"
    assert vp.kanji_signature(old) == vp.kanji_signature(new) == "私海"
    # 句読点・カタカナ・ラテン文字も落ちる
    assert vp.kanji_signature("ファルス（FARCE）に就て") == "就"


@pytest.mark.unit
def test_t208_jaccard_basics():
    """T-208 / F-06: Jaccard の境界。空集合で 0、同一で 1。"""
    assert vp.jaccard(set(), {"あ"}) == 0.0
    assert vp.jaccard({"あ", "い"}, {"あ", "い"}) == 1.0
    assert vp.jaccard({"あ", "い"}, {"い", "う"}) == pytest.approx(1 / 3)


@pytest.mark.validation
def test_t209_threshold_sits_in_an_empty_gap():
    """T-209 / F-06: 閾値は実測の谷にある。

    較正の抽出出力(2026-08-25、32,352 組比較):
      一致群の最小 0.7597 / 非一致群の最大 0.0112
    閾値を定数で正当化するのではなく、**その間に 1 組も存在しない**ことを主張する。
    谷が埋まったら(新規収録などで)このテストが落ち、較正のやり直しを促す。
    """
    d = _pairs()
    cal = d["provenance"]["calibration"]
    assert cal["matched_min"] > 10 * cal["unmatched_max"], (
        f"分布が二峰でなくなった: 一致最小 {cal['matched_min']} / 非一致最大 {cal['unmatched_max']}"
    )
    assert cal["unmatched_max"] < d["provenance"]["threshold"] < cal["matched_min"]


@pytest.mark.validation
def test_t210_titles_alone_would_miss_pairs():
    """T-210 / F-06: 題名一致では二重版を確定できない(HC-012 の実証)。

    抽出出力(2026-08-25)に含まれる題名違いのペア:
      石の思ひ ⇔ 石の思い / いづこへ ⇔ いずこへ / をみな ⇔ おみな / 歴史と現実 ⇔ 歴史と事実
    逆に同題名の連作(安吾巷談・明治開化 安吾捕物)はペアに入らない。
    """
    d = _pairs()
    pairs = d["pairs"]
    diff = {(p["title_a"], p["title_b"]) for p in pairs if not p["same_title"]}
    for a, b in [("石の思ひ", "石の思い"), ("いづこへ", "いずこへ"), ("をみな", "おみな")]:
        assert (a, b) in diff or (b, a) in diff, f"{a}⇔{b} が検出されていない"
    # 連作どうしが二重版として誤検出されていない
    for p in pairs:
        if p["pair_type"] == "variant":
            assert p["kana_a"] != p["kana_b"]
            assert not (
                p["title_a"].startswith("安吾巷談 ") and p["title_b"].startswith("安吾巷談 ")
            )


@pytest.mark.validation
def test_t211_no_card_in_two_pairs():
    """T-211 / F-06: 1 つのカードが 2 つ以上のペアに属さない。"""
    d = _pairs()
    seen = {}
    for p in d["pairs"]:
        for cid in (p["a"], p["b"]):
            assert cid not in seen, f"{cid} が複数のペアに属する: {seen.get(cid)} と {p}"
            seen[cid] = p
