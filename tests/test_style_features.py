"""文体特徴量の抽出(F-08)。

抽出器はまず**手計算フィクスチャで較正**してから全展開する。
定義をテスト側に明文化し、実装がその定義に従うことを確かめる(導出をコメントに残す)。
"""
import pytest

from pipeline import style_features as sf


# 手計算用の最小テキスト。全角空白は行頭の字下げ。
HAND = "　私は行く。彼は「そうだ」と言った。"
# 文字の内訳(手で数える):
#   字下げの全角空白 1 を除くと 17 字 — 私 は 行 く 。 彼 は 「 そ う だ 」 と 言 っ た 。
#   期待値は裸の数値ではなく**列挙**で書く(HC-019(b): 読んで検算できる形にする)
# 文は 2 つ:
#   s1 = 私は行く。            5 字
#   s2 = 彼は「そうだ」と言った。 12 字


@pytest.mark.unit
def test_t301_sentence_split_and_length():
    """T-301 / F-08: 文の分割と文長。句点で切り、行頭の字下げは長さに数えない。"""
    ss = sf.sentences(HAND)
    assert ss == ["私は行く。", "彼は「そうだ」と言った。"]
    assert [len(s) for s in ss] == [5, 12]
    f = sf.features(HAND)
    assert f["sent_len_mean"] == pytest.approx(8.5)
    # 分散は母分散(n で割る): ((5-8.5)^2 + (12-8.5)^2)/2 = 12.25
    assert f["sent_len_var"] == pytest.approx(12.25)


@pytest.mark.unit
def test_t302_char_class_ratios():
    """T-302 / F-08: 文字種比率。分母は字下げを除いた本文字数 17。

    漢字 = 私行彼言、平仮名 = はくはそうだとった(「は」が 2 回あることに注意)。
    数え間違いを読者が検算できるよう、期待値は文字列の長さで書く。
    """
    f = sf.features(HAND)
    assert f["n_chars"] == 17
    assert f["kanji_ratio"] == pytest.approx(len("私行彼言") / 17)
    assert f["katakana_ratio"] == pytest.approx(0.0)
    assert f["hiragana_ratio"] == pytest.approx(len("はくはそうだとった") / 17)


@pytest.mark.unit
def test_t303_punctuation_and_speech():
    """T-303 / F-08: 句読点密度(1000 字あたり)と会話文比率。

    句点 = 。。 の 2 個、読点 0。鉤括弧の中身 = そうだ の 3 字。
    """
    f = sf.features(HAND)
    assert f["kuten_per_1000"] == pytest.approx(len("。。") / 17 * 1000)
    assert f["touten_per_1000"] == pytest.approx(0.0)
    assert f["punct_per_1000"] == pytest.approx(len("。。") / 17 * 1000)
    assert f["speech_ratio"] == pytest.approx(len("そうだ") / 17)


@pytest.mark.unit
def test_t304_speech_ratio_unclosed():
    """T-304 / F-08: 閉じ括弧が無い会話は行末までとみなす(定義を固定する)。"""
    # 全体 = 「ああ + 次の行。、会話部分 = ああ(開き括弧は会話に数えない)
    f = sf.features("「ああ" + chr(10) + "次の行。")
    assert f["n_chars"] == len("「ああ") + len("次の行。")
    assert f["speech_ratio"] == pytest.approx(len("ああ") / f["n_chars"])


@pytest.mark.unit
def test_t305_pos_features_on_folded_text():
    """T-305 / F-08: 品詞・助詞助動詞の特徴は**畳んだ表現**の上で取る。

    実測(2026-08-25、二重版 12 組と無関係 30 組):
      原文    版間 L1 0.0218 / 作品間 L1 0.1099 → S/N  5.0
      fold    版間 L1 0.0023 / 作品間 L1 0.1057 → S/N 45.8
    畳むと作品間の判別力はほぼ変わらないまま、版間の差だけが 9 分の 1 になる。
    """
    old = "私は海をだきしめてゐたい。思ふに、さうしてしまふ。"
    new = "私は海をだきしめていたい。思うに、そうしてしまう。"
    fo, fn = sf.features(old), sf.features(new)
    for k in fo:
        if k.startswith(("pos_", "func_")):
            assert fo[k] == pytest.approx(fn[k]), f"{k} が表記で変わる"


@pytest.mark.validation
def test_t306_feature_distribution_sanity():
    """T-306 / F-08: 全件に値を作った直後の分布検査(HC-022)。

    退化値(NaN・無限・全件同値)が無いこと。単体テストは境界を含まないので、
    集約の誤りはここでしか露見しない。
    """
    import json
    import math
    from pathlib import Path as P

    fp = P(__file__).resolve().parents[1] / "data" / "style_features.json"
    if not fp.exists():
        pytest.skip("style_features.json 未生成")
    d = json.loads(fp.read_text(encoding="utf-8"))
    feats = d["features"]
    for cid, f in feats.items():
        for k, v in f.items():
            assert math.isfinite(v), f"{cid} の {k} が有限でない: {v}"
    for k in d["keys"]:
        vals = {f.get(k, 0.0) for f in feats.values()}
        assert len(vals) > 1, f"特徴量 {k} が全件同値(退化している)"


@pytest.mark.validation
def test_t307_chunk_average_matches_whole_text():
    """T-307 / F-08: 部分の集約と全体の直接計算が一致する(HC-022)。

    比率型の特徴量は、チャンク平均が全文一括の値から大きく離れてはならない。
    末尾の極小チャンクを等重みで混ぜていた欠陥はこの検査で捕まる
    (card45744 で平仮名率 0.607 → 0.406 に歪んでいた)。
    """
    from pathlib import Path as P

    from pipeline import aozora_parser as ap

    raw = P(__file__).resolve().parents[1] / "data" / "raw"
    targets = ["45744", "45828", "42620", "42618"]
    files = [raw / f"{c}.txt" for c in targets]
    if not all(p.exists() for p in files):
        pytest.skip("data/raw が未取得")
    for p in files:
        with open(p, encoding="utf-8", newline="") as fh:
            body = ap.parse(fh.read()).body_text
        chunked = sf.features(body)
        whole = sf.features(body, chunk=10**9)
        for k in ("kanji_ratio", "hiragana_ratio", "speech_ratio"):
            assert abs(chunked[k] - whole[k]) < 0.05, (
                f"{p.stem} の {k}: チャンク平均 {chunked[k]:.4f} と全文 {whole[k]:.4f} が乖離"
            )
