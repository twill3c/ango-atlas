"""青空文庫記法パーサー(F-03)と再直列化(F-04)。

設計上の要点:
- **往復検査が最上位の制約**。parse → serialize は原文と 1 バイトも違ってはならない。
  そのため全トークンは原文断片を復元できる情報を持つ(改行コードも変換しない)。
- ヘッダ/フッタの境界は実測に基づく:
    ヘッダ = 先頭〜「記号について」罫線ブロックの終端(罫線が無い作品は題名・著者ブロックまで)
    フッタ = 「底本：」で始まる行以降
- ルビの自動ベースは、《 の直前から**同じ文字種の連続**を遡って取る。
  ｜ がある場合はそこまでを明示ベースとする。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

RULE_RE = re.compile(r"^-{10,}$")
FOOTER_RE = re.compile(r"^底本[：:]")

# 文字種(自動ルビのベース決定用)
_KANJI = r"一-鿿々〆ヶヵ豈-﫿㐀-䶿"
_HIRA = r"ぁ-ゖゝゞ"
_KATA = r"ァ-ヴーヽヾ"
_LATIN = r"0-9A-Za-z０-９Ａ-Ｚａ-ｚ"
_CLASSES = (_KANJI, _HIRA, _KATA, _LATIN)


@dataclass
class Text:
    raw: str

    def out(self) -> str:
        return self.raw


@dataclass
class Ruby:
    """ルビ。base は原文の生断片で、外字注記 ［＃…］ を含みうる(実測)。"""

    base: str
    ruby: str
    explicit: bool = False

    @property
    def base_text(self) -> str:
        """読み本文用のベース。注記を落とす(※ は 1 文字として残る)。"""
        return re.sub(r"［＃[^］]*］", "", self.base)

    def out(self) -> str:
        return ("｜" if self.explicit else "") + self.base + "《" + self.ruby + "》"


@dataclass
class Annotation:
    """入力者注 ［＃…］。verbatim 保存する(解釈は上位レイヤの仕事)。"""

    raw: str

    def out(self) -> str:
        return self.raw


Node = Text | Ruby | Annotation


@dataclass
class Doc:
    title: str
    author: str
    header: str
    body: list[Node]
    footer: str
    subtitle: list[str] = field(default_factory=list)
    header_raw: str = ""
    footer_raw: str = ""

    @property
    def body_text(self) -> str:
        """注記・ルビ記号を除いた読み本文(ルビはベースのみ残す)。"""
        out = []
        for n in self.body:
            if isinstance(n, Text):
                out.append(n.raw)
            elif isinstance(n, Ruby):
                out.append(n.base_text)
        return "".join(out)

    def rubies(self) -> list[Ruby]:
        return [n for n in self.body if isinstance(n, Ruby)]

    def annotations(self) -> list[Annotation]:
        return [n for n in self.body if isinstance(n, Annotation)]


def _class_of(ch: str) -> str | None:
    for cls in _CLASSES:
        if re.match(f"[{cls}]", ch):
            return cls
    return None


def _auto_base(prefix: str) -> str:
    """《 の直前から自動でベースを決める。

    実測(2026-08-24、docs/notation_inventory.md)で判明した型:
        高尚なること※［＃「木＋解」、第3水準1-86-22］《かしわ》   → base=※［＃…］
        僕の顳※［＃「需＋頁」、第3水準1-94-6］《こめかみ》        → base=顳※［＃…］
        屡※［＃二の字点、1-2-22］《しばしば》                     → base=屡※［＃…］
    外字注記・二の字点は 1 文字の漢字の代替なので、ベースに取り込み、
    その手前は**漢字類のみ**遡る(直前が仮名なら外字 1 文字だけがベース)。
    """
    base = ""
    saw_note = False
    while True:
        m = re.search(r"※?［＃[^］]*］$", prefix)
        if not m:
            break
        base = m.group(0) + base
        prefix = prefix[: m.start()]
        saw_note = True
    if saw_note:
        cls = _KANJI
    else:
        if not prefix:
            return base
        cls = _class_of(prefix[-1])
        if cls is None:
            return base
    m2 = re.search(f"[{cls}]+$", prefix)
    return (m2.group(0) if m2 else "") + base


_NOTE_RE = re.compile(r"［＃[^］]*］")
_RUBY_RE = re.compile(r"《[^》]*》")


def parse_body_only(body: str) -> Doc:
    """本文だけをトークン化する(ヘッダ/フッタを持たない断片用)。"""
    return Doc(title="", author="", header="", body=_tokenize(body), footer="")


def _flush(raw: str, nodes: list) -> None:
    """未確定テキストを Text / Annotation に切って積む。"""
    pos = 0
    for m in _NOTE_RE.finditer(raw):
        if m.start() > pos:
            nodes.append(Text(raw[pos : m.start()]))
        nodes.append(Annotation(m.group(0)))
        pos = m.end()
    if pos < len(raw):
        nodes.append(Text(raw[pos:]))


def _tokenize(body: str) -> list[Node]:
    """ルビを軸にトークン化する。

    注記はベースの一部になりうる(外字・二の字点)ため、ルビ確定まで未確定テキストの
    中に生のまま保持し、ベースを切り出したあとで Text / Annotation に分解する。
    """
    nodes: list[Node] = []
    pending = ""
    pos = 0
    for m in _RUBY_RE.finditer(body):
        pending += body[pos : m.start()]
        ruby = m.group(0)[1:-1]
        # ｜ は同一行内のものだけを明示ベースとみなす
        bar = pending.rfind("｜")
        if bar >= 0 and chr(10) not in pending[bar:]:
            base = pending[bar + 1 :]
            pending = pending[:bar]
            explicit = True
        else:
            base = _auto_base(pending)
            pending = pending[: len(pending) - len(base)]
            explicit = False
        _flush(pending, nodes)
        pending = ""
        nodes.append(Ruby(base=base, ruby=ruby, explicit=explicit))
        pos = m.end()
    pending += body[pos:]
    _flush(pending, nodes)
    return nodes


def split_sections(src: str) -> tuple[str, str, str]:
    """(ヘッダ, 本文, フッタ) に分ける。境界は行単位・原文の改行を保持する。"""
    # 行に分けるが、改行文字は各行の末尾に残す(往復のため)
    lines = re.findall(r"[^\n]*\n|[^\n]+$", src)

    def bare(s: str) -> str:
        return s.rstrip("\r\n")

    rules = [i for i, l in enumerate(lines) if RULE_RE.match(bare(l))]
    if len(rules) >= 2:
        header_end = rules[1] + 1
    else:
        # 罫線ブロックが無い作品: 題名・著者ブロック(先頭の非空行群)の直後まで
        header_end = 0
        seen_text = False
        for i, l in enumerate(lines):
            if bare(l).strip():
                seen_text = True
            elif seen_text:
                header_end = i + 1
                break
    footer_start = next(
        (i for i, l in enumerate(lines) if FOOTER_RE.match(bare(l))), len(lines)
    )
    if footer_start < header_end:  # 異常な並び。本文なしとして扱う
        header_end = footer_start
    return (
        "".join(lines[:header_end]),
        "".join(lines[header_end:footer_start]),
        "".join(lines[footer_start:]),
    )


def _head_fields(header: str) -> tuple[str, list[str], str]:
    """題名・副題・著者を取る。

    実測(2026-08-24): 先頭の空行までが題名ブロックで、題名 → (副題) → 著者名 の順。
    副題を持つ例 card42816「阿部定という女 / （浅田一博士へ） / 坂口安吾」があるため、
    著者はブロックの 2 行目ではなく**最終行**を取る。
    """
    block: list[str] = []
    for line in header.split(chr(10)):
        bare = line.rstrip(chr(13))
        if not bare.strip():
            if block:
                break
            continue
        block.append(bare)
    if not block:
        return "", [], ""
    if len(block) == 1:
        return block[0], [], ""
    return block[0], block[1:-1], block[-1]


def parse(src: str) -> Doc:
    header, body, footer = split_sections(src)
    title, subtitle, author = _head_fields(header)
    return Doc(
        title=title,
        author=author,
        subtitle=subtitle,
        header=header,
        body=_tokenize(body),
        footer=footer,
    )


def serialize(doc: Doc) -> str:
    return doc.header + "".join(n.out() for n in doc.body) + doc.footer
