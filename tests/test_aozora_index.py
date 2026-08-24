"""青空文庫 作家ページ・カードページの解析(F-01)。

期待値の出所: tests/fixtures/PROVENANCE.md に記した実測データ(2026-08-24 取得)。
件数は定数で書かず、ページ自身の <li> 集合との一致で検証する(HC-016)。
"""
import re
from pathlib import Path

import pytest

from pipeline import aozora_index as ix

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def author_html():
    return (FIX / "person1095.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def card_html():
    return (FIX / "card42877.html").read_text(encoding="utf-8")


@pytest.mark.unit
def test_t006_works_are_bijective_with_page_list(author_html):
    """T-006 / F-01: 公開中セクションの <li> と抽出結果が全単射。

    期待値は定数ではなく「ページ自身が持つ <li> 集合との一致」という不変量。
    """
    i = author_html.find('name="sakuhin_list_1"')
    j = author_html.find('name="sakuhin_list_2"')
    assert 0 < i < j, "公開中/作業中のアンカーが見つからない"
    page_cards = set(re.findall(r"card(\d+)\.html", author_html[i:j]))

    works = ix.parse_author_page(author_html)
    got = {w["card_id"] for w in works}

    assert got == page_cards, f"取りこぼし {page_cards - got} / 余分 {got - page_cards}"
    assert len(works) == len(got), "card_id の重複がある"


@pytest.mark.unit
def test_t006b_every_work_has_required_fields(author_html):
    """T-006 / F-01: 全 entry に card_url・文字遣い種別・題名がある。"""
    works = ix.parse_author_page(author_html)
    for w in works:
        assert re.fullmatch(
            r"https://www\.aozora\.gr\.jp/cards/\d{6}/card\d+\.html", w["card_url"]
        ), w["card_url"]
        assert w["kana_type"] in {"新字新仮名", "新字旧仮名", "旧字旧仮名", "旧字新仮名"}
        assert w["title"].strip(), f"題名が空: {w['card_id']}"


@pytest.mark.unit
def test_t007_title_includes_subtitle_outside_anchor(author_html):
    """T-007 / F-01: 連作の副題は <a> の外にある。題名は再構成しなければならない。

    実測(2026-08-24): card43172 の行は
      <a ...>安吾巷談</a>　01 麻薬・自殺・宗教（新字新仮名、作品ID：43172）
    アンカーテキストだけでは 12 件の「安吾巷談」が区別できない。
    """
    by_card = {w["card_id"]: w for w in ix.parse_author_page(author_html)}
    assert by_card["43172"]["title"] == "安吾巷談 01 麻薬・自殺・宗教"
    assert by_card["43203"]["title"] == "明治開化 安吾捕物 01 読者への口上"
    # 副題が全角ダッシュのもの(実測: card42864)
    assert by_card["42864"]["title"] == "教祖の文学 ――小林秀雄論――"
    # 題名内の丸括弧が文字遣いの括弧と紛れないこと(実測 2026-08-24: card42816 の原文は
    #   <a>阿部定という女</a>　（浅田一博士へ）（新字新仮名、作品ID：42816）
    # で、アンカーと副題は全角空白で区切られている。連結時は空白 1 個に正規化する)
    assert by_card["42816"]["title"] == "阿部定という女 （浅田一博士へ）"
    assert by_card["42816"]["kana_type"] == "新字新仮名"


@pytest.mark.unit
def test_t008_titles_alone_do_not_identify_variant_pairs(author_html):
    """T-008 / F-01・F-06: 同題名は二重版とは限らない(連作が大量にある)。

    L1 では「題名一致 = 二重版」と確定してはならない(HC-012)。ここでは
    「アンカーテキストだけの同名が連作を含む」ことを実測として固定する。
    """
    works = ix.parse_author_page(author_html)
    by_card = {w["card_id"]: w for w in works}
    # 実測: 安吾巷談 12 件はすべて新字新仮名 = 二重版ではない
    kouden = [w for w in works if w["title"].startswith("安吾巷談 ")]
    assert len(kouden) > 1
    assert {w["kana_type"] for w in kouden} == {"新字新仮名"}
    # 実測: 風博士 は 新字新仮名(42616) と 新字旧仮名(43024) の対 = 二重版候補
    assert by_card["42616"]["title"] == by_card["43024"]["title"] == "風博士"
    assert {by_card["42616"]["kana_type"], by_card["43024"]["kana_type"]} == {
        "新字新仮名",
        "新字旧仮名",
    }


@pytest.mark.unit
def test_t009_card_page_fields(card_html):
    """T-009 / F-02: カードページから本文 zip・初出・文字遣い・NDC を取る。

    期待値は card42877.html の実測(2026-08-24)。
    """
    meta = ix.parse_card_page(card_html, card_id="42877")
    assert meta["ruby_zip_url"] == (
        "https://www.aozora.gr.jp/cards/001095/files/42877_ruby_27472.zip"
    )
    assert meta["kana_type"] == "新字新仮名"
    assert meta["ndc"] == "NDC 913"
    assert meta["shoshutsu"] == "「愛と美」朝日新聞社、1947（昭和22）年10月5日"


@pytest.mark.unit
def test_t010_translation_card_lives_under_another_person(author_html):
    """T-010 / F-01: 安吾の作家ページには他作家ディレクトリのカードが混ざる。

    実測(2026-08-24): 公開中 513 件のうち card45791「〔翻訳〕ステファヌ・マラルメ」
    のみが person 001217(ヴァレリー)配下で、安吾は訳者である。本文は翻訳文なので
    文体解析での扱いは別途判断が要る — L1 では事実の記録に留める(HC-012)。
    件数ではなく「own_work=False の集合」という不変量で書く。
    """
    works = ix.parse_author_page(author_html)
    foreign = [w for w in works if not w["own_work"]]
    assert {w["card_id"] for w in foreign} == {"45791"}
    (w,) = foreign
    assert w["person_id"] == "001217"
    assert w["other_author"] == {"name": "ヴァレリー ポール", "role": "著者"}
    assert w["card_url"] == "https://www.aozora.gr.jp/cards/001217/card45791.html"
    # 安吾自身の著作は全て 001095 配下
    assert {x["person_id"] for x in works if x["own_work"]} == {"001095"}
