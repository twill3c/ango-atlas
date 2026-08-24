"""作品メタデータの構築(F-05)。

出所は 2 系統ある:
  1. カードページの「初出」欄(data/aozora_works.json の shoshutsu)
  2. 本文フッタの「初出：」行
実測(2026-08-25)では 498 件が両方を持ち、1 件だけ西暦が食い違う(card45737、青空文庫側の
誤植)。元号は両者で一致するため、**元号と整合する側を採る**。判断の根拠は evidence に残す。

ジャンルは NDC 分類と連作題名からのみ決める。NDC 914 は随筆と評論を区別しないので、
統制語彙も「随筆・評論」に統合する(区別できない語彙を立てない — HC-012)。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline import aozora_parser as ap

ROOT = Path(__file__).resolve().parents[1]
WORKS = ROOT / "data" / "aozora_works.json"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "works_meta.json"

GENRES = ("小説", "随筆・評論", "探偵小説", "巷談・ルポ", "紀行・日記", "戯曲")

# 元号の元年(明治1=1868 / 大正1=1912 / 昭和1=1926)
_ERA_BASE = {"明治": 1867, "大正": 1911, "昭和": 1925}
_ERA_RE = re.compile(r"(明治|大正|昭和)([0-9０-９〇一二三四五六七八九十百]+)")
_AD_RE = re.compile(r"(1[89]\d\d)")
_KANJI_NUM = {c: i for i, c in enumerate("〇一二三四五六七八九")}

# NDC の主分類 → ジャンル。複合表記は先頭の 3 桁を見る
_NDC_GENRE = {
    "913": "小説",
    "914": "随筆・評論",
    "915": "紀行・日記",
    "912": "戯曲",
    "910": "随筆・評論",
    "901": "随筆・評論",
    "816": "随筆・評論",
    "795": "随筆・評論",
    "796": "随筆・評論",
    "779": "随筆・評論",
    "775": "随筆・評論",
}
# 連作は NDC より優先する(実測で確認した系列のみ書く)
_SERIES_GENRE = {
    "明治開化 安吾捕物": "探偵小説",
    "明治開化 安吾捕物帖": "探偵小説",
    "安吾巷談": "巷談・ルポ",
    "安吾人生案内": "巷談・ルポ",
    "安吾新日本風土記": "紀行・日記",
    "安吾の新日本地理": "紀行・日記",
    "安吾史譚": "随筆・評論",
}
_SERIES_RE = re.compile(r"^(?P<name>.+?)\s(?P<no>\d{2})\s")


def _to_int(s: str) -> int | None:
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if s.isdigit():
        return int(s)
    # 十/二十三 のような表記
    if "十" in s:
        a, _, b = s.partition("十")
        tens = _KANJI_NUM.get(a, 1) if a else 1
        ones = _KANJI_NUM.get(b, 0) if b else 0
        return tens * 10 + ones
    if len(s) == 1 and s in _KANJI_NUM:
        return _KANJI_NUM[s]
    return None


def gengo_year(era: str, num: str) -> int | None:
    n = _to_int(num)
    return None if n is None else _ERA_BASE[era] + n


def year_of(s: str | None) -> tuple[int | None, int | None]:
    """初出文字列から (西暦表記の年, 元号から換算した年) を返す。"""
    if not s:
        return (None, None)
    ad = _AD_RE.search(s)
    era = _ERA_RE.search(s)
    return (
        int(ad.group(1)) if ad else None,
        gengo_year(era.group(1), era.group(2)) if era else None,
    )


def resolve_year(card: str | None, footer: str | None) -> dict:
    """2 出所から初出年を決める。食い違いは元号と整合する側を採る。"""
    cand = [("card", card), ("footer", footer)]
    parsed = {k: year_of(v) for k, v in cand}
    years = {k: (ad if ad is not None else era) for k, (ad, era) in parsed.items()}
    have = {k: y for k, y in years.items() if y is not None}
    if not have:
        ev = card or footer or ""
        return {
            "pub_year": None,
            "source": None,
            "conflict": False,
            "evidence": ev,
        }
    conflict = len(set(have.values())) > 1
    if conflict:
        # 元号と西暦が一致する側を正とする
        for k in ("card", "footer"):
            ad, era = parsed[k]
            if ad is not None and era is not None and ad == era:
                text = card if k == "card" else footer
                return {
                    "pub_year": ad,
                    "source": k,
                    "conflict": True,
                    "evidence": f"{text}(他方と西暦が食い違うが元号と整合)",
                }
        # どちらも整合しない場合は元号を優先する
        for k in ("card", "footer"):
            _, era = parsed[k]
            if era is not None:
                text = card if k == "card" else footer
                return {
                    "pub_year": era,
                    "source": k,
                    "conflict": True,
                    "evidence": f"{text}(西暦が不整合のため元号を採用)",
                }
    k = "card" if "card" in have else "footer"
    text = card if k == "card" else footer
    return {"pub_year": have[k], "source": k, "conflict": False, "evidence": text}


def series_of(title: str) -> tuple[str | None, int | None]:
    m = _SERIES_RE.match(title)
    if not m:
        return (None, None)
    return (m.group("name"), int(m.group("no")))


def genre_of(ndc: str | None, title: str) -> tuple[str | None, str]:
    """(ジャンル, 根拠) を返す。決められないときは (None, 'needs_review')。"""
    name, _ = series_of(title)
    if name and name in _SERIES_GENRE:
        return (_SERIES_GENRE[name], f"連作『{name}』")
    if ndc:
        for code in re.findall(r"\d{3}", ndc):
            if code in _NDC_GENRE:
                return (_NDC_GENRE[code], f"{ndc}")
    return (None, "needs_review")


_FOOT_SHO = re.compile(r"初出：(.*?)(?=\n[^　\s]|\Z)", re.S)


def footer_shoshutsu(footer: str) -> str | None:
    m = _FOOT_SHO.search(footer)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def build() -> list[dict]:
    works = json.loads(WORKS.read_text(encoding="utf-8"))["works"]
    out = []
    for w in works:
        with open(RAW / f"{w['card_id']}.txt", encoding="utf-8", newline="") as f:
            doc = ap.parse(f.read())
        y = resolve_year(w.get("shoshutsu"), footer_shoshutsu(doc.footer))
        genre, gsrc = genre_of(w.get("ndc"), w["title"])
        name, no = series_of(w["title"])
        out.append(
            {
                "card_id": w["card_id"],
                "work_id": w["work_id"],
                "title": w["title"],
                "kana_type": w["kana_type"],
                "own_work": w["own_work"],
                "text_kind": w.get("text_kind"),
                "chars": w.get("chars"),
                "ndc": w.get("ndc"),
                "series": name,
                "series_no": no,
                "genre": genre,
                "genre_source": gsrc,
                "pub_year": y["pub_year"],
                "pub_year_source": y["source"],
                "pub_year_conflict": y["conflict"],
                "pub_year_evidence": y["evidence"],
            }
        )
    return out


if __name__ == "__main__":
    import collections

    recs = build()
    OUT.write_text(
        json.dumps(
            {
                "provenance": {
                    "built_from": ["data/aozora_works.json", "data/raw/*.txt"],
                    "built_at": "2026-08-25",
                    "note": "初出年はカード欄と本文フッタの 2 出所。食い違いは元号で判定",
                },
                "works": recs,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    g = collections.Counter(r["genre"] for r in recs)
    print("ジャンル:", g.most_common())
    print("年不明:", sum(1 for r in recs if r["pub_year"] is None),
          "/ 食い違い:", sum(1 for r in recs if r["pub_year_conflict"]))
    yr = [r["pub_year"] for r in recs if r["pub_year"]]
    print(f"年の範囲: {min(yr)}–{max(yr)}  → {OUT}")
