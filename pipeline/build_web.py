"""web/ 用データの生成(L1 時点はコーパス実測サマリのみ)。"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKS = ROOT / "data" / "aozora_works.json"
OUT = ROOT / "web" / "data" / "corpus_summary.json"


def build() -> dict:
    doc = json.loads(WORKS.read_text(encoding="utf-8"))
    works = doc["works"]
    got = [w for w in works if not w.get("external_host")]
    kana = collections.Counter(w["kana_type"] for w in works)
    ndc = collections.Counter(
        (w.get("ndc") or "不明").replace("NDC ", "") for w in got
    )
    chars = sum(w.get("chars", 0) for w in got)
    return {
        "provenance": doc["provenance"],
        "listed": len(works),
        "fetched": len(got),
        "unavailable": [
            {"card_id": w["card_id"], "title": w["title"], "reason": w.get("evidence")}
            for w in works
            if w.get("external_host")
        ],
        "translated_cards": [
            {"card_id": w["card_id"], "title": w["title"], "other_author": w["other_author"]}
            for w in works
            if not w.get("own_work")
        ],
        "kana_type": dict(kana.most_common()),
        "ndc_top": dict(ndc.most_common(8)),
        "chars_total": chars,
        "chars_max": max((w.get("chars", 0) for w in got), default=0),
        "chars_min": min((w.get("chars", 0) for w in got), default=0),
        "longest": sorted(
            ({"title": w["title"], "chars": w.get("chars", 0)} for w in got),
            key=lambda x: -x["chars"],
        )[:5],
    }


if __name__ == "__main__":
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"listed={d['listed']} fetched={d['fetched']} chars={d['chars_total']:,} → {OUT}")
