"""仮名遣いの折りたたみ(F-07)。

規則は二重版 50 組の差分の実測から導出した(`docs/kana_fold_calibration.md`)。
テストの期待値も実測出力から貼る(HC-019)。
"""
import json
from pathlib import Path

import pytest

from pipeline import kana_fold as kf

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "data" / "fold_calibration.json"


@pytest.mark.unit
def test_t212_variant_spellings_fold_to_the_same_string():
    """T-212 / F-07: 同じ内容の新旧仮名が同一表現に畳まれる。

    抽出出力(2026-08-25):
      fold('私は海をだきしめてゐたい') == fold('私は海をだきしめていたい') == '私わ海おだきしめていたい'
      fold('思ふに、さうしてしまふ')   == fold('思うに、そうしてしまう')   == '思うに、そうしてしもう'
      fold('けふのてふてふ')           == fold('きょうのちょうちょう')     == 'きようのちようちよう'
    助詞の は/を も畳むが、両版に同じ関数を掛けるので比較では対称に効く。
    """
    assert kf.fold("私は海をだきしめてゐたい") == kf.fold("私は海をだきしめていたい")
    assert kf.fold("思ふに、さうしてしまふ") == kf.fold("思うに、そうしてしまう")
    assert kf.fold("けふのてふてふ") == kf.fold("きょうのちょうちょう")
    assert kf.fold("私は海をだきしめてゐたい") == "私わ海おだきしめていたい"


@pytest.mark.unit
def test_t213_order_hagyo_before_choon():
    """T-213 / F-07: ハ行転呼を長音より先に適用する。

    旧「しまふ」→「しまう」→「しもう」、新「しまう」→「しもう」。
    逆順だと新側だけが「しもう」になり差が開く(実測でこの順が中央値を上げた)。
    """
    assert kf.fold("してしまふ") == kf.fold("してしまう")
    # 長音規則そのものは効いている(素通しではない)
    assert kf.fold_choon("さう") == "そう"
    assert kf.fold_choon("せう") == "しよう"


@pytest.mark.unit
def test_t214_odoriji_expansion():
    """T-214 / F-07: 踊り字の展開。

    ゝ は直前字の繰り返し、ゞ は濁点付き、くの字点 ／＼ は直前 2 字の繰り返し。
    """
    assert kf.expand_odoriji("こゝろ") == "こころ"
    assert kf.expand_odoriji("たゞし") == "ただし"
    assert kf.expand_odoriji("人々") == "人人"
    assert kf.expand_odoriji("わざ／＼") == "わざわざ"
    assert kf.expand_odoriji("さん／″＼") == "さんざん"


@pytest.mark.validation
def test_t215_folding_improves_every_variant_pair():
    """T-215 / F-07・O-1: 二重版 50 組すべてで一致率が上がる。

    較正出力(2026-08-25): 素 中央値 0.9488 → 畳んだ後 0.9964、≥0.99 が 0/50 → 41/50。
    定数の閾値ではなく「**どの組も悪化しない**」「中央値が上がる」という不変量で書く。
    残差は仮名遣いではなく底本差なので、1.0 を要求してはならない。
    """
    if not CAL.exists():
        pytest.skip("fold_calibration.json 未生成")
    d = json.loads(CAL.read_text(encoding="utf-8"))
    worse = [p for p in d["per_pair"] if p["folded"] < p["raw"]]
    assert not worse, f"畳んで悪化したペア: {worse[:5]}"
    assert d["stages"]["g4_choon_full_fold"]["median"] > d["stages"]["raw"]["median"]
    assert d["stages"]["g4_choon_full_fold"]["min"] > d["stages"]["raw"]["min"]
    # 規則群は累積で単調に効いている(どの段でも中央値が下がらない)
    med = [v["median"] for v in d["stages"].values()]
    assert med == sorted(med), f"途中で中央値が下がる規則がある: {med}"
