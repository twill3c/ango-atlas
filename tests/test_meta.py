"""作品メタデータの構築(F-05)。

期待値の出所: data/aozora_works.json と data/raw の実測(2026-08-25)。
抽出出力を貼る(HC-019)。件数は集合の不変量で書く(HC-016)。
"""
from pathlib import Path

import pytest

from pipeline import build_meta as bm

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_t201_gengo_to_seireki():
    """T-201 / F-05: 元号 → 西暦の換算。

    期待値の出所: 元号の定義(明治1=1868 / 大正1=1912 / 昭和1=1926)。
    """
    assert bm.gengo_year("明治", "1") == 1868
    assert bm.gengo_year("大正", "1") == 1912
    assert bm.gengo_year("昭和", "1") == 1926
    assert bm.gengo_year("昭和", "17") == 1942
    assert bm.gengo_year("昭和", "三十") == 1955


@pytest.mark.unit
def test_t202_year_from_shoshutsu():
    """T-202 / F-05: 初出文字列からの年抽出。西暦と元号の両方を見る。

    実測の書式(2026-08-25):
      '「愛と美」朝日新聞社、1947（昭和22）年10月5日'
      '「西日本新聞 第二四九四六号～第二五〇四六号」1953（昭和28）年1月2日～4月13日'
      '「神港夕刊新聞」「九州タイムズ」発表年月日未詳'
    """
    assert bm.year_of("「愛と美」朝日新聞社、1947（昭和22）年10月5日") == (1947, 1947)
    # 期間表記は最初の年を取る
    assert bm.year_of("「西日本新聞」1953（昭和28）年1月2日～4月13日") == (1953, 1953)
    # 年月日未詳
    assert bm.year_of("「神港夕刊新聞」「九州タイムズ」発表年月日未詳") == (None, None)
    # 西暦が誤植で元号と食い違う実例(card45737 のフッタ)
    assert bm.year_of("「都新聞」 1982（昭和17）年9月30日号") == (1982, 1942)


@pytest.mark.unit
def test_t203_conflict_resolved_by_gengo():
    """T-203 / F-05: 出所が食い違うとき、元号と整合する側を採る。

    実測(2026-08-25): card45737「今日の感想」はカードが 1942（昭和17）、
    本文フッタが 1982（昭和17）。昭和17 = 1942 なのでフッタの西暦が誤植である。
    青空文庫側の誤植なので、判断の根拠を evidence に残して記録する。
    """
    r = bm.resolve_year(
        card="「都新聞」1942（昭和17）年9月30日",
        footer="「都新聞」 1982（昭和17）年9月30日号",
    )
    assert r["pub_year"] == 1942
    assert r["conflict"] is True
    assert r["source"] == "card"
    assert "昭和" in r["evidence"]


@pytest.mark.unit
def test_t204_series_split():
    """T-204 / F-05: 連作の題名から系列名と番号を取る。

    実測(2026-08-25)の題名:
      '安吾巷談 01 麻薬・自殺・宗教'
      '明治開化 安吾捕物 07 その六 血を見る真珠'
      '青鬼の褌を洗う女'(連作でない)
    """
    assert bm.series_of("安吾巷談 01 麻薬・自殺・宗教") == ("安吾巷談", 1)
    assert bm.series_of("明治開化 安吾捕物 07 その六 血を見る真珠") == ("明治開化 安吾捕物", 7)
    assert bm.series_of("青鬼の褌を洗う女") == (None, None)


@pytest.mark.unit
def test_t205_genre_vocabulary_and_evidence():
    """T-205 / F-05: ジャンルは統制語彙内で、必ず根拠を持つ。

    NDC 914 は随筆と評論を区別しない(実測 262 件)。区別できない語彙は立てない。
    連作は NDC より優先する(明治開化 安吾捕物 = 探偵小説、安吾巷談 = 巷談・ルポ)。
    """
    assert bm.genre_of("NDC 913", "青鬼の褌を洗う女")[0] == "小説"
    assert bm.genre_of("NDC 914", "堕落論")[0] == "随筆・評論"
    assert bm.genre_of("NDC 915", "安吾の新日本地理 01")[0] == "紀行・日記"
    assert bm.genre_of("NDC 913", "明治開化 安吾捕物 07 その六")[0] == "探偵小説"
    assert bm.genre_of("NDC 914", "安吾巷談 01 麻薬・自殺・宗教")[0] == "巷談・ルポ"
    # NDC が無いカードは推定で埋めない
    genre, source = bm.genre_of(None, "咢堂小論")
    assert genre is None and source == "needs_review"
    for ndc in ("NDC 913", "NDC 914", "NDC 915", None):
        g, _ = bm.genre_of(ndc, "題名")
        assert g in bm.GENRES or g is None


@pytest.mark.validation
def test_t206_meta_covers_corpus():
    """T-206 / F-05: メタデータが全作品を覆い、必須欄を満たす。

    件数ではなく aozora_works.json との集合一致で書く(HC-016)。
    """
    meta_path = ROOT / "data" / "works_meta.json"
    if not meta_path.exists():
        pytest.skip("works_meta.json 未生成")
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))["works"]
    src = json.loads(
        (ROOT / "data" / "aozora_works.json").read_text(encoding="utf-8")
    )["works"]
    assert {m["card_id"] for m in meta} == {w["card_id"] for w in src}
    for m in meta:
        assert m["genre"] in bm.GENRES or m["genre"] is None
        assert m["pub_year"] is None or 1900 < m["pub_year"] < 1960
        if m["genre"] is None:
            assert m["genre_source"] == "needs_review"
        if m["pub_year"] is not None:
            assert m["pub_year_evidence"]
