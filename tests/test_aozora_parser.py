"""青空文庫記法パーサーと往復検査(F-03 / F-04)。

期待値の出所: tests/fixtures/PROVENANCE.md(実データ、2026-08-24 取得)。
往復検査は自己完結オラクル — 原文そのものが正解であり外部正解を要さない。
"""
from pathlib import Path

import pytest

from pipeline import aozora_parser as ap

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def excerpt() -> str:
    # 改行コードは原文どおり CRLF。newline="" で変換させない
    with open(FIX / "aoonino_excerpt.txt", encoding="utf-8", newline="") as f:
        return f.read()


@pytest.mark.unit
def test_t001_header_body_footer(excerpt):
    """T-001 / F-03: 罫線までがヘッダ、底本：以降がフッタ。"""
    doc = ap.parse(excerpt)
    assert doc.title == "青鬼の褌を洗う女"
    assert doc.author == "坂口安吾"
    assert "【テキスト中に現れる記号について】" in doc.header
    assert doc.footer.startswith("底本：「坂口安吾全集")
    assert "底本" not in doc.body_text
    assert "【テキスト中に現れる記号について】" not in doc.body_text
    # 本文冒頭は実測(HC-019: 目視ではなく抽出出力を貼る)。
    #   python -c "...; print(ap.parse(src).body_text.splitlines()[:2])"
    #   → ['', '\u3000匂いって何だろう？']
    assert doc.body_text.splitlines()[:2] == ['', '\u3000匂いって何だろう？']


@pytest.mark.unit
def test_t002_ruby_auto_base(excerpt):
    """T-002 / F-03: ルビの自動ベース取り。

    期待値は抽出出力から貼る(HC-019):
      python -c "...; print(sorted((r.base,r.ruby,r.explicit) for r in ap.parse(src).rubies()))"
      → [('噛', 'かじ', False), ('米', 'メートル', True)]

    ヘッダの記号説明ブロックにも 3 件の 《》 があるが(（例）倅《せがれ》 等)、
    それらは本文のルビとして数えてはならない。この分離自体が検証項目である。
    """
    doc = ap.parse(excerpt)
    assert sorted((r.base, r.ruby, r.explicit) for r in doc.rubies()) == [
        ("噛", "かじ", False),
        ("米", "メートル", True),
    ]
    assert "《" in doc.header, "ヘッダの用例が失われている(分離の前提が壊れた)"
    for r in doc.rubies():
        assert r.base, f"ベースが空のルビがある: 《{r.ruby}》"


@pytest.mark.unit
def test_t003_explicit_base_marker(excerpt):
    """T-003 / F-03: ｜ による明示ベース。再直列化で ｜ が復元される。

    実測: 「百｜米《メートル》」 → base=米、explicit=True
    """
    doc = ap.parse(excerpt)
    explicit = [r for r in doc.rubies() if r.explicit]
    assert explicit, "｜ 付きルビが検出されていない"
    assert ("米", "メートル") in {(r.base, r.ruby) for r in explicit}
    assert "｜米《メートル》" in ap.serialize(doc)


@pytest.mark.unit
def test_t004_annotation_verbatim():
    """T-004 / F-03: 入力者注 ［＃…］ は verbatim 保存される。

    実測(2026-08-24、堕落論 card42620 本文): ※［＃「虫＋廷」、第4水準2-87-52］
    ここでは記法の最小形を直接与える(原文断片の再現)。
    """
    src = "　蜩《ひぐらし》の※［＃「虫＋廷」、第4水準2-87-52］が鳴く。\r\n"
    doc = ap.parse_body_only(src)
    notes = [n.raw for n in doc.annotations()]
    assert notes == ["［＃「虫＋廷」、第4水準2-87-52］"]
    assert ap.serialize(doc) == src


@pytest.mark.unit
def test_t005_roundtrip_excerpt(excerpt):
    """T-005 / F-04: serialize(parse(x)) == x(全文一致)。"""
    assert ap.serialize(ap.parse(excerpt)) == excerpt


@pytest.mark.validation
def test_t101_roundtrip_all_raw():
    """T-101 / F-04: 取得済み全テキストの往復。不一致 0 件(100%)。"""
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    files = sorted(raw.glob("*.txt"))
    if not files:
        pytest.skip("data/raw が未取得(fetch_aozora を先に実行する)")
    bad = []
    for p in files:
        with open(p, encoding="utf-8", newline="") as f:
            src = f.read()
        if ap.serialize(ap.parse(src)) != src:
            bad.append(p.name)
    assert not bad, f"往復不一致 {len(bad)}/{len(files)} 件: {bad[:10]}"


@pytest.mark.unit
def test_t011_gaiji_note_as_ruby_base():
    """T-011 / F-03: 外字注記・二の字点はルビのベースになる。

    実測(2026-08-24、docs/notation_inventory.md で 9 件検出。抽出出力を貼る — HC-019):
      card45797  base='顳※［＃「需＋頁」、第3水準1-94-6］'  base_text='顳※'
      card57479  base='屡※［＃二の字点、1-2-22］'          base_text='屡※'
      card42616  base='※［＃「木＋解」、第3水準1-86-22］'   base_text='※'
    直前が仮名のとき(card42616「こと」の後)は外字 1 文字だけがベースになる。
    """
    cases = [
        ("僕の顳※［＃「需＋頁」、第3水準1-94-6］《こめかみ》", "顳※［＃「需＋頁」、第3水準1-94-6］", "顳※"),
        ("屡※［＃二の字点、1-2-22］《しばしば》", "屡※［＃二の字点、1-2-22］", "屡※"),
        ("高尚なること※［＃「木＋解」、第3水準1-86-22］《かしわ》", "※［＃「木＋解」、第3水準1-86-22］", "※"),
    ]
    for src, base, base_text in cases:
        doc = ap.parse_body_only(src)
        (r,) = doc.rubies()
        assert r.base == base
        assert r.base_text == base_text
        assert ap.serialize(doc) == src, "往復が壊れた"


@pytest.mark.validation
def test_t102_no_empty_ruby_base_in_corpus():
    """T-102 / F-03: 取得済み全文でベースが空のルビが存在しない。

    件数ではなく「空ベースの集合が空」という不変量(HC-016)。
    """
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    files = sorted(raw.glob("*.txt"))
    if not files:
        pytest.skip("data/raw が未取得")
    bad = []
    for p in files:
        with open(p, encoding="utf-8", newline="") as f:
            doc = ap.parse(f.read())
        bad += [(p.stem, r.ruby) for r in doc.rubies() if not r.base]
    assert bad == [], f"ベースが空のルビ: {bad[:10]}"
