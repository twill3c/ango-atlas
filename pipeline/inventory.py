"""コーパス実測インベントリ(F-03 の設計根拠 / F-04 の全件検査)。

パーサーの分岐は「推定」ではなく本インベントリの実測に基づいて決める。
出力は docs/notation_inventory.md と標準出力。
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from pipeline import aozora_parser as ap

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "docs" / "notation_inventory.md"

RULE_RE = re.compile(r"^-{10,}$")


def read(p: Path) -> str:
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def analyze() -> dict:
    files = sorted(RAW.glob("*.txt"))
    notes = collections.Counter()
    note_shapes = collections.Counter()
    rule_counts = collections.Counter()
    footer_missing: list[str] = []
    roundtrip_bad: list[str] = []
    empty_body: list[str] = []
    empty_base: list[tuple[str, str]] = []
    ruby_total = 0
    for p in files:
        src = read(p)
        lines = [l.rstrip("\r\n") for l in src.split("\n")]
        rule_counts[sum(1 for l in lines if RULE_RE.match(l))] += 1
        if not any(l.startswith("底本") for l in lines):
            footer_missing.append(p.stem)
        doc = ap.parse(src)
        if ap.serialize(doc) != src:
            roundtrip_bad.append(p.stem)
        if not doc.body_text.strip():
            empty_body.append(p.stem)
        for n in doc.annotations():
            notes[n.raw] += 1
            # 「…に傍点」「中見出し」など、注記の型を粗く集計する
            body = n.raw[2:-1]
            shape = re.sub(r"「[^」]*」", "「…」", body)
            shape = re.sub(r"\d+", "N", shape)
            note_shapes[shape] += 1
        for r in doc.rubies():
            ruby_total += 1
            if not r.base:
                empty_base.append((p.stem, r.ruby))
    return {
        "files": len(files),
        "roundtrip_bad": roundtrip_bad,
        "footer_missing": footer_missing,
        "empty_body": empty_body,
        "rule_counts": dict(sorted(rule_counts.items())),
        "ruby_total": ruby_total,
        "ruby_empty_base": empty_base,
        "note_total": sum(notes.values()),
        "note_shapes": note_shapes.most_common(40),
    }


def render(d: dict) -> str:
    L = [
        "# 注記・記法インベントリ(実測)",
        "",
        f"対象: `data/raw/*.txt` {d['files']} 件。パーサーの分岐はこの実測に基づく。",
        "",
        "## 往復検査(F-04)",
        "",
        f"- 不一致: **{len(d['roundtrip_bad'])} 件** {d['roundtrip_bad'][:20]}",
        f"- 本文が空: {len(d['empty_body'])} 件 {d['empty_body'][:20]}",
        f"- 「底本」行が無いファイル: {len(d['footer_missing'])} 件 {d['footer_missing'][:20]}",
        "",
        "## ヘッダ罫線の本数分布",
        "",
        "| 罫線本数 | ファイル数 |",
        "|---|---|",
    ]
    L += [f"| {k} | {v} |" for k, v in d["rule_counts"].items()]
    L += [
        "",
        "## ルビ",
        "",
        f"- 総数: {d['ruby_total']}",
        f"- ベースが空: {len(d['ruby_empty_base'])} 件 {d['ruby_empty_base'][:20]}",
        "",
        "## 入力者注の型(上位 40)",
        "",
        f"総数 {d['note_total']}。「…」は引用部、N は数字を潰した形。",
        "",
        "| 型 | 件数 |",
        "|---|---|",
    ]
    L += [f"| `{k}` | {v} |" for k, v in d["note_shapes"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    d = analyze()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(d), encoding="utf-8")
    print(json.dumps({k: v for k, v in d.items() if k != "note_shapes"},
                     ensure_ascii=False)[:900])
    print(f"→ {OUT}")
