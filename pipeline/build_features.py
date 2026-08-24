"""全作品の文体特徴量を作る(F-08)。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from pipeline import aozora_parser as ap
from pipeline import style_features as sf

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "works_meta.json"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "style_features.json"


def build(fold_pos: bool = True) -> dict:
    meta = json.loads(META.read_text(encoding="utf-8"))["works"]
    feats = {}
    t0 = time.time()
    for i, r in enumerate(meta, 1):
        cid = r["card_id"]
        with open(RAW / f"{cid}.txt", encoding="utf-8", newline="") as f:
            body = ap.parse(f.read()).body_text
        feats[cid] = sf.features(body, fold_pos=fold_pos)
        if i % 100 == 0:
            print(f"  {i}/{len(meta)} ({time.time()-t0:.0f}s)")
    keys = sorted({k for f in feats.values() for k in f})
    return {
        "provenance": {
            "built_at": "2026-08-25",
            "note": "品詞・機能語は kana_fold.fold 上で取る(docs/style_features_calibration.md)",
            "chunk": sf.CHUNK,
        },
        "keys": keys,
        "features": feats,
    }


if __name__ == "__main__":
    import argparse

    a = argparse.ArgumentParser()
    a.add_argument("--no-fold-pos", action="store_true",
                   help="較正の対照条件: 品詞を原文のまま取る")
    ns = a.parse_args()
    d = build(fold_pos=not ns.no_fold_pos)
    d["provenance"]["fold_pos"] = not ns.no_fold_pos
    out = OUT if not ns.no_fold_pos else OUT.with_name("style_features_rawpos.json")
    out.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    OUT_ = out
    print(f"{len(d['features'])} 作品 × {len(d['keys'])} 特徴量 → {out}")
