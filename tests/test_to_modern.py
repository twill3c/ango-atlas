"""可読な現代仮名遣いへの寄せ(F-14 の入力条件)。

`fold` が比較用の破壊的な畳み込みなのに対し、`to_modern` は埋め込みモデルに与える
自然な日本語を作る。完全な変換ではないので、期待値も「読める形になる」ことに絞る。
効果の判定はテストではなく二重版チャンク対応の実測で行う(docs/embed_eval.md)。
"""
import pytest

from pipeline import kana_fold as kf


@pytest.mark.unit
def test_t408_to_modern_keeps_particles():
    """T-408 / F-14: 助詞の は/へ/を を壊さない(fold との決定的な違い)。

    抽出出力(2026-08-25):
      to_modern('私は海をだきしめてゐたい') → '私は海をだきしめていたい'
      fold      (同上)                      → '私わ海おだきしめていたい'
    """
    assert kf.to_modern("私は海をだきしめてゐたい") == "私は海をだきしめていたい"
    assert kf.to_modern("東京へ行かう") == "東京へ行こう"
    assert kf.fold("私は海をだきしめてゐたい") == "私わ海おだきしめていたい"


@pytest.mark.unit
def test_t409_to_modern_youon_and_hagyo():
    """T-409 / F-14: 拗音・長音と、漢字直後のハ行転呼。

    抽出出力(2026-08-25):
      'てふてふが飛ぶ'        → 'ちょうちょうが飛ぶ'
      'きやうだいは三人ゐる'  → 'きょうだいは三人いる'
      'かなしい幸ひ'          → 'かなしい幸い'
      '思ふに、けふはさうしてしまふ' → '思うに、きょうはそうしてしまう'
    """
    assert kf.to_modern("てふてふが飛ぶ") == "ちょうちょうが飛ぶ"
    assert kf.to_modern("きやうだいは三人ゐる") == "きょうだいは三人いる"
    assert kf.to_modern("かなしい幸ひ") == "かなしい幸い"
    assert kf.to_modern("思ふに、けふはさうしてしまふ") == "思うに、きょうはそうしてしまう"


@pytest.mark.unit
def test_t410_to_modern_leaves_modern_text_alone():
    """T-410 / F-14: すでに現代仮名遣いの文はほとんど変えない。

    旧仮名版だけでなく新仮名版にも同じ関数を掛けるので、副作用が小さいことが要る。
    """
    for s in ["私は海をだきしめていたい。", "半年のうちに世相は変った。", "そういうものだろう。"]:
        assert kf.to_modern(s) == s


@pytest.mark.validation
def test_t411_to_modern_improves_variant_agreement():
    """T-411 / F-14: 二重版チャンク対応で、素より一致率が上がる。

    実測(2026-08-25、1,026 組): 中央値 0.9474 → 0.9714、改善 1,001 組 / 悪化 16 組。
    定数ではなく「中央値が上がる」「悪化が改善より少ない」という不変量で書く。
    """
    import difflib
    import json
    import statistics
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cp = root / "data" / "chunks.json"
    ep = root / "data" / "eval_variant_chunks.json"
    if not (cp.exists() and ep.exists()):
        pytest.skip("チャンク/対応表が未生成")
    ch = {c["i"]: c for c in json.loads(cp.read_text(encoding="utf-8"))["chunks"]}
    ev = json.loads(ep.read_text(encoding="utf-8"))["pairs"][:300]
    raw, mod = [], []
    for e in ev:
        o, n = ch[e["query_chunk"]]["text"], ch[e["gold_chunk"]]["text"]
        raw.append(difflib.SequenceMatcher(None, o, n, autojunk=False).ratio())
        mod.append(difflib.SequenceMatcher(None, kf.to_modern(o), n, autojunk=False).ratio())
    assert statistics.median(mod) > statistics.median(raw)
    worse = sum(1 for a, b in zip(raw, mod) if b < a)
    better = sum(1 for a, b in zip(raw, mod) if b > a)
    assert worse < better / 10, f"悪化 {worse} / 改善 {better}"
