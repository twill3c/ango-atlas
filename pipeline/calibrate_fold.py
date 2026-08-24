"""折りたたみ表現の効果測定(F-07 の較正)。

二重版 50 組を唯一のオラクルとして、規則群ごとの一致率を測る。
出力: data/fold_calibration.json(テストが読む) と docs/kana_fold_calibration.md。
"""
from __future__ import annotations

import difflib
import json
import statistics
from pathlib import Path

from pipeline import aozora_parser as ap
from pipeline import kana_fold as kf

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "variant_pairs.json"
RAW = ROOT / "data" / "raw"
OUT_JSON = ROOT / "data" / "fold_calibration.json"
OUT_DOC = ROOT / "docs" / "kana_fold_calibration.md"


def body(cid: str) -> str:
    with open(RAW / f"{cid}.txt", encoding="utf-8", newline="") as f:
        return ap.parse(f.read()).body_text


def variant_pairs() -> list[tuple[str, str, str]]:
    """(旧仮名 card, 新仮名 card, 題名) の一覧。"""
    d = json.loads(PAIRS.read_text(encoding="utf-8"))
    out = []
    for p in d["pairs"]:
        if p["pair_type"] != "variant":
            continue
        old = p["a"] if p["kana_a"] == "新字旧仮名" else p["b"]
        new = p["b"] if old == p["a"] else p["a"]
        out.append((old, new, p["title_a"]))
    return out


_STAGES = {
    "raw": lambda s: s,
    "g1_odoriji": kf.expand_odoriji,
    "g2_small": lambda s: kf.expand_odoriji(s).translate(kf._SMALL),
    "g3_kyuji": lambda s: kf.expand_odoriji(s).translate(kf._SMALL).translate(kf._KYUJI),
    "g5_hagyo": lambda s: kf.expand_odoriji(s)
    .translate(kf._SMALL)
    .translate(kf._KYUJI)
    .translate(kf._HAGYO),
    "g4_choon_full_fold": kf.fold,
}


def run() -> dict:
    pairs = variant_pairs()
    texts = {cid: body(cid) for pair in pairs for cid in pair[:2]}
    stages = {}
    per_pair = []
    for name, fn in _STAGES.items():
        ratios = []
        for old, new, _t in pairs:
            r = difflib.SequenceMatcher(
                None, fn(texts[old]), fn(texts[new]), autojunk=False
            ).ratio()
            ratios.append(r)
        stages[name] = {
            "median": round(statistics.median(ratios), 4),
            "min": round(min(ratios), 4),
            "ge_099": sum(1 for r in ratios if r >= 0.99),
        }
        if name in ("raw", "g4_choon_full_fold"):
            for (old, new, t), r in zip(pairs, ratios):
                if name == "raw":
                    per_pair.append({"old": old, "new": new, "title": t, "raw": round(r, 4)})
                else:
                    next(p for p in per_pair if p["old"] == old)["folded"] = round(r, 4)
    return {
        "provenance": {"pairs_from": "data/variant_pairs.json", "built_at": "2026-08-25"},
        "n_pairs": len(pairs),
        "stages": stages,
        "per_pair": per_pair,
    }


if __name__ == "__main__":
    d = run()
    OUT_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [
        "# 仮名遣い折りたたみの較正(実測)",
        "",
        f"オラクル: 二重版 {d['n_pairs']} 組(`data/variant_pairs.json`)。",
        "同一作品の新字新仮名版と新字旧仮名版に同じ関数を掛け、本文の一致率を測る。",
        "",
        "| 規則群(累積) | 一致率 中央値 | 最小 | ≥0.99 |",
        "|---|---|---|---|",
    ]
    label = {
        "raw": "素(変換なし)",
        "g1_odoriji": "+ G1 踊り字展開",
        "g2_small": "+ G2 小書き→大書き",
        "g3_kyuji": "+ G3 ゐゑぢづ",
        "g5_hagyo": "+ G5 ハ行転呼・を",
        "g4_choon_full_fold": "+ G4 長音(= fold 全体)",
    }
    for k, v in d["stages"].items():
        lines.append(f"| {label[k]} | {v['median']} | {v['min']} | {v['ge_099']}/{d['n_pairs']} |")
    worst = sorted(d["per_pair"], key=lambda p: p["folded"])[:5]
    lines += [
        "",
        "## 残差が大きいペア",
        "",
        "| 作品 | 素 | 畳んだ後 |",
        "|---|---|---|",
    ] + [f"| {p['title']} | {p['raw']} | {p['folded']} |" for p in worst] + [
        "",
        "残差は仮名遣いではなく**底本差**(咒/呪・言/云・馬鹿の表記・括弧の種別・句読点)である。",
        "正規化で詰められる範囲は尽きており、これ以上の規則追加は過適合になる。",
        "",
        "## 適用順",
        "",
        "ハ行転呼(G5)は長音(G4)より**先**に適用する。旧「しまふ」→「しまう」→「しもう」、",
        "新「しまう」→「しもう」で一致する。逆順だと新側だけが変換され、かえって差が開く。",
        "",
    ]
    OUT_DOC.write_text("\n".join(lines), encoding="utf-8")
    for k, v in d["stages"].items():
        print(f"{label[k]:22s} 中央値 {v['median']} 最小 {v['min']} ≥0.99 {v['ge_099']}/{d['n_pairs']}")
    print(f"→ {OUT_JSON} / {OUT_DOC}")
