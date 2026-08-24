"""青空文庫 坂口安吾コーパスの取得(F-01 / F-02)。

N-02: 取得間隔 1 秒以上・User-Agent 明示・再実行はキャッシュ優先(HTTP を出さない)。
N-03: 全データに取得元 URL と取得日を記録する。

使い方:
    python -m pipeline.fetch_aozora            # 全件(キャッシュ優先)
    python -m pipeline.fetch_aozora --limit 5  # 先頭 5 件だけ
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

from pipeline import aozora_index as ix

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
WORKS_JSON = ROOT / "data" / "aozora_works.json"

AUTHOR_URL = f"{ix.BASE}/index_pages/person{int(ix.PERSON_ID)}.html"
UA = "ango-atlas/0.1 (research; https://github.com/twill3c/ango-atlas)"
MIN_INTERVAL = 1.0  # 秒(N-02)

_last_request = 0.0


def _today() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date().isoformat()


def fetch(url: str, cache_name: str) -> bytes:
    """キャッシュ優先の取得。キャッシュがあれば HTTP を出さない(N-02)。"""
    global _last_request
    path = CACHE / cache_name
    if path.exists():
        return path.read_bytes()
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    _last_request = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def decode_html(body: bytes) -> str:
    for enc in ("utf-8", "cp932", "euc_jp"):
        try:
            return body.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("HTML の文字コードを判定できない")


def decode_aozora_text(body: bytes) -> tuple[str, str]:
    """青空文庫本文の復号。JIS X 0208/ShiftJIS が基本だが実測で例外がありうる。"""
    for enc in ("cp932", "shift_jis_2004", "utf-8", "euc_jp"):
        try:
            return body.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError("本文の文字コードを判定できない")


def extract_text(zip_bytes: bytes) -> tuple[str, str, str]:
    """zip から .txt を 1 件取り出す。戻り値 (本文, zip 内ファイル名, 判定した符号化)。"""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = [n for n in z.namelist() if n.lower().endswith(".txt")]
    if len(names) != 1:
        raise ValueError(f"zip 内の .txt が 1 件でない: {names}")
    text, enc = decode_aozora_text(z.read(names[0]))
    return text, names[0], enc


def run(limit: int | None = None) -> list[dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    author_html = decode_html(fetch(AUTHOR_URL, f"person{int(ix.PERSON_ID)}.html"))
    works = ix.parse_author_page(author_html)
    if limit:
        works = works[:limit]

    records: list[dict] = []
    for n, w in enumerate(works, 1):
        rec = dict(w)
        rec["source"] = {"author_page": AUTHOR_URL, "fetched_at": _today()}
        try:
            card_html = decode_html(fetch(w["card_url"], f"card{w['card_id']}.html"))
            meta = ix.parse_card_page(
                card_html, card_id=w["card_id"], person_id=w["person_id"]
            )
            rec.update(
                {k: meta[k] for k in ("ruby_zip_url", "ndc", "shoshutsu", "note")}
            )
            # カードページ側の文字遣い種別と作家ページ側の食い違いは記録して残す
            if meta["kana_type"] and meta["kana_type"] != w["kana_type"]:
                rec["kana_type_mismatch"] = meta["kana_type"]
        except Exception as e:  # noqa: BLE001 — 取得不能の理由を残すのが目的
            rec["external_host"] = True
            rec["evidence"] = f"カードページ取得/解析に失敗: {type(e).__name__}: {e}"
            records.append(rec)
            print(f"[{n}/{len(works)}] {w['card_id']} カード失敗: {e}")
            continue

        if not rec["ruby_zip_url"]:
            rec["external_host"] = True
            rec["evidence"] = "カードページにルビあり zip の掲載が無い(実測)"
            records.append(rec)
            print(f"[{n}/{len(works)}] {w['card_id']} ルビ zip 無し: {w['title']}")
            continue

        try:
            zname = rec["ruby_zip_url"].rsplit("/", 1)[-1]
            text, inner, enc = extract_text(fetch(rec["ruby_zip_url"], zname))
        except Exception as e:  # noqa: BLE001
            rec["external_host"] = True
            rec["evidence"] = f"本文取得に失敗: {type(e).__name__}: {e}"
            records.append(rec)
            print(f"[{n}/{len(works)}] {w['card_id']} 本文失敗: {e}")
            continue

        out = RAW / f"{w['card_id']}.txt"
        # 改行コードは原文どおり(往復検査 F-04 の前提)
        out.write_text(text, encoding="utf-8", newline="")
        rec.update(
            {
                "external_host": False,
                "text_file": f"data/raw/{out.name}",
                "text_zip_inner": inner,
                "source_encoding": enc,
                "chars": len(text),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "fetched_at": _today(),
            }
        )
        records.append(rec)
        if n % 25 == 0 or n == len(works):
            print(f"[{n}/{len(works)}] {w['card_id']} {w['title'][:24]} ({len(text)} 字)")

    WORKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    WORKS_JSON.write_text(
        json.dumps(
            {
                "provenance": {
                    "author_page": AUTHOR_URL,
                    "fetched_at": _today(),
                    "note": "件数は作家ページ『公開中の作品』の実測。定数で書かない(HC-016)",
                },
                "works": records,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    recs = run(a.limit)
    ok = [r for r in recs if not r.get("external_host")]
    print(f"完了: {len(recs)} 件中 本文取得 {len(ok)} 件 → {WORKS_JSON}")
