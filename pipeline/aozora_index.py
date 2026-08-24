"""青空文庫 作家ページ・カードページの解析(F-01 / F-02)。

実測主義: 取得したページの構造を実測して書いている。推定で確定しない(HC-012)。
- 公開中の作品リストは <h2><a name="sakuhin_list_1"> と sakuhin_list_2 の間にある
- 連作の副題は <a> の外側にある(例: <a>安吾巷談</a>　01 麻薬・自殺・宗教（新字新仮名、…）)
- 文字遣い種別は行末の括弧内。題名自身が丸括弧を含む例があるため、末尾側から取る
"""
from __future__ import annotations

import re

PERSON_ID = "001095"
BASE = "https://www.aozora.gr.jp"
KANA_TYPES = ("新字新仮名", "新字旧仮名", "旧字旧仮名", "旧字新仮名")

_LI = re.compile(r"<li>(.*?)</li>", re.S)
_ENTRY = re.compile(
    r'<a href="\.\./cards/(?P<person>\d+)/card(?P<card>\d+)\.html">(?P<anchor>.*?)</a>'
    r"(?P<rest>.*)",
    re.S,
)
# 末尾の「（文字遣い、作品ID：NNNN）」。題名内の丸括弧と紛れないよう、この形にだけ一致させる
_TAIL = re.compile(
    r"（(?P<kana>" + "|".join(KANA_TYPES) + r")、作品ID：(?P<wid>\d+)）"
)
# 他作家のカードに安吾が別役割で関与する行(実測 2026-08-24: card45791 の 1 件のみ)
#   〔翻訳〕ステファヌ・マラルメ（新字旧仮名、作品ID：45791）　→ヴァレリー ポール(著者)
_OTHER_ROLE = re.compile(r"→(?P<name>[^(（]+)[(（](?P<role>[^)）]+)[)）]")


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def _normalize(s: str) -> str:
    """全角空白・連続空白を半角 1 個に潰し、前後を落とす(題名の再構成用)。"""
    return re.sub(r"[\s　]+", " ", s).strip()


def _section(html: str, start_anchor: str, end_anchor: str | None) -> str:
    i = html.find(f'name="{start_anchor}"')
    if i < 0:
        raise ValueError(f"アンカーが見つからない: {start_anchor}")
    j = html.find(f'name="{end_anchor}"') if end_anchor else len(html)
    if j < 0:
        j = len(html)
    return html[i:j]


def parse_author_page(html: str, section: str = "sakuhin_list_1") -> list[dict]:
    """作家ページの作品リストを構造化する。

    戻り値の各要素: card_id / work_id / title / kana_type / card_url
    題名は「アンカーテキスト + アンカー外の副題」を連結して再構成する。
    """
    end = "sakuhin_list_2" if section == "sakuhin_list_1" else None
    seg = _section(html, section, end)
    out: list[dict] = []
    for li in _LI.findall(seg):
        m = _ENTRY.search(li)
        if not m:
            raise ValueError(f"想定外の <li> 構造: {_normalize(_strip_tags(li))[:80]}")
        rest = _strip_tags(m.group("rest"))
        tail = _TAIL.search(rest)
        if not tail:
            raise ValueError(f"文字遣い種別が取れない: {_normalize(rest)[:80]}")
        subtitle = _normalize(rest[: tail.start()])
        anchor = _normalize(_strip_tags(m.group("anchor")))
        title = f"{anchor} {subtitle}".strip() if subtitle else anchor
        card_id = m.group("card")
        person = m.group("person")
        other = _OTHER_ROLE.search(rest[tail.end():])
        out.append(
            {
                "card_id": card_id,
                "work_id": tail.group("wid"),
                "title": title,
                "kana_type": tail.group("kana"),
                "card_url": f"{BASE}/cards/{person}/card{card_id}.html",
                "person_id": person,
                # 安吾自身の著作か(person ディレクトリが安吾のものか)。
                # 他作家カードの場合、安吾は訳者等であり本文は翻訳文である —
                # 文体解析での扱いは L2/L3 で判断する(除外を L1 で決め打ちしない)
                "own_work": person == PERSON_ID,
                "other_author": (
                    {"name": _normalize(other.group("name")), "role": other.group("role")}
                    if other
                    else None
                ),
            }
        )
    return out


_CARD_ROW = re.compile(
    r'<td class="header">(?P<key>[^<]*?)：?</td>\s*<td>(?P<val>.*?)</td>', re.S
)
_RUBY_ZIP = re.compile(r'href="\./files/(?P<name>\d+_ruby_\d+\.zip)"')
# ルビあり版が無いカードがある(実測 2026-08-24: 513 件中 86 件)。本文自体は取得できるので
# ルビなし版へフォールバックし、どちらを取ったかを text_kind に残す(F-02)
_TXT_ZIP = re.compile(r'href="\./files/(?P<name>\d+_txt_\d+\.zip)"')


def parse_card_page(html: str, card_id: str, person_id: str = PERSON_ID) -> dict:
    """カードページから本文 zip の URL と作品データを取る。

    ルビあり zip が無いカード(外部ホスト等)では ruby_zip_url が None になる。
    その場合の扱いは呼び出し側で external_host 等として明示する(F-02)。
    """
    # 「分類」は作品データ表にも作家データ表にもある(作家側は「著者」)。
    # 作品データ表の範囲に限定しないと、作品データに分類が無いカードで作家側を拾う
    i = html.find("作品データ")
    j = html.find("作家データ", i + 1) if i >= 0 else -1
    work_table = html[i:j] if i >= 0 and j > i else html
    rows: dict[str, str] = {}
    for m in _CARD_ROW.finditer(work_table):
        key = _normalize(_strip_tags(m.group("key")))
        if key and key not in rows:
            rows[key] = _normalize(_strip_tags(m.group("val")))
    zm = _RUBY_ZIP.search(html)
    tm = _TXT_ZIP.search(html)
    ruby_url = f"{BASE}/cards/{person_id}/files/{zm.group('name')}" if zm else None
    txt_url = f"{BASE}/cards/{person_id}/files/{tm.group('name')}" if tm else None
    return {
        "card_id": card_id,
        "ruby_zip_url": ruby_url,
        "text_zip_url": ruby_url or txt_url,
        "text_kind": "ruby" if ruby_url else ("noruby" if txt_url else None),
        "kana_type": rows.get("文字遣い種別"),
        "ndc": rows.get("分類"),
        "shoshutsu": rows.get("初出"),
        "note": rows.get("備考"),
    }
