"""web/ 用データの生成(L1 時点はコーパス実測サマリのみ)。"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKS = ROOT / "data" / "aozora_works.json"
META = ROOT / "data" / "works_meta.json"
PAIRS = ROOT / "data" / "variant_pairs.json"
CAL = ROOT / "data" / "fold_calibration.json"
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
    extra = {}
    if META.exists():
        m = json.loads(META.read_text(encoding="utf-8"))["works"]
        years = [r["pub_year"] for r in m if r["pub_year"]]
        extra["genre"] = dict(
            collections.Counter(r["genre"] or "不明" for r in m).most_common()
        )
        extra["year_range"] = [min(years), max(years)]
        extra["year_unknown"] = sum(1 for r in m if r["pub_year"] is None)
        extra["by_year"] = dict(sorted(collections.Counter(years).items()))
    if PAIRS.exists():
        pr = json.loads(PAIRS.read_text(encoding="utf-8"))
        extra["pairs"] = {
            "variant": sum(1 for p in pr["pairs"] if p["pair_type"] == "variant"),
            "duplicate": sum(1 for p in pr["pairs"] if p["pair_type"] == "duplicate"),
            "calibration": pr["provenance"]["calibration"],
        }
    if CAL.exists():
        c = json.loads(CAL.read_text(encoding="utf-8"))
        extra["fold"] = {
            "raw_median": c["stages"]["raw"]["median"],
            "folded_median": c["stages"]["g4_choon_full_fold"]["median"],
        }
    return {
        **extra,
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
