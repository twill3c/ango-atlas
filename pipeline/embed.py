"""チャンク埋め込みの生成(F-14)。

プロジェクト専用の venv(.venv)で実行する。共有 venv には torch を入れない。
    .venv/Scripts/python.exe -m pipeline.embed --model ruri

モデルごとの作法(モデルカードの実測 2026-08-25):
  cl-nagoya/ruri-base          前置き「クエリ: 」「文章: 」、平均プーリング、768 次元
  intfloat/multilingual-e5-base 前置き「query: 」「passage: 」、平均プーリング、768 次元
どちらも L2 正規化して内積=コサイン類似度にする。

入力の作り方は 2 通りを比べられる:
  raw     原文のまま(旧仮名は旧仮名のまま)
  modern  可読な現代仮名遣いへ寄せる(kana_fold.to_modern)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pipeline import chunks as ck
from pipeline import kana_fold as kf

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.json"
META = ROOT / "data" / "works_meta.json"
OUTDIR = ROOT / "data" / "embeddings"

# 実測 2026-08-25(8 コア CPU、96 チャンクの所要から全 15,430 件を外挿):
#   cl-nagoya/ruri-v3-30m           10 分  256 次元  平均プーリング  前置き 検索クエリ:/検索文書:
#   intfloat/multilingual-e5-small  17 分  384 次元  平均プーリング  前置き query:/passage:
#   cl-nagoya/ruri-base            195 分  768 次元  ← この CPU では非現実的なので採らない
#   cl-nagoya/ruri-small           トークナイザが transformers 5.x で読めず断念
MODELS = {
    "ruri": {"repo": "cl-nagoya/ruri-v3-30m", "query": "検索クエリ: ", "passage": "検索文書: "},
    "e5": {"repo": "intfloat/multilingual-e5-small", "query": "query: ", "passage": "passage: "},
    "ruri-base": {"repo": "cl-nagoya/ruri-base", "query": "クエリ: ", "passage": "文章: "},
}


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    h = last_hidden.masked_fill(~mask[..., None].bool(), 0.0)
    return h.sum(dim=1) / mask.sum(dim=1)[..., None]


class Encoder:
    def __init__(self, name: str):
        self.spec = MODELS[name]
        self.tok = AutoTokenizer.from_pretrained(self.spec["repo"])
        self.model = AutoModel.from_pretrained(self.spec["repo"])
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], kind: str = "passage", batch: int = 16,
               max_length: int = 512) -> np.ndarray:
        prefix = self.spec[kind]
        out = []
        for i in range(0, len(texts), batch):
            part = [prefix + t for t in texts[i : i + batch]]
            enc = self.tok(part, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
            res = self.model(**enc)
            v = mean_pool(res.last_hidden_state, enc["attention_mask"])
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            out.append(v.to(torch.float32).numpy())
        return np.concatenate(out, axis=0)


def passages(mode: str, context: bool = True, subset: set[int] | None = None
             ) -> tuple[list[str], list[int]]:
    """埋め込む文字列と、対応するチャンク番号。

    context=False は較正用の対照条件。作品名・ジャンル・初出年の前置きを付けずに
    本文だけを埋め込む(小さいモデルでは前置きが本文を押しのける恐れがあるため)。
    subset を渡すと、そのチャンク番号だけを対象にする(部分実験用)。
    """
    meta = {r["card_id"]: r for r in json.loads(META.read_text(encoding="utf-8"))["works"]}
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))["chunks"]
    texts, ids = [], []
    for c in chunks:
        if subset is not None and c["i"] not in subset:
            continue
        r = meta[c["card_id"]]
        body = c["text"] if mode == "raw" else kf.to_modern(c["text"])
        texts.append(
            ck.with_context({"text": body}, r["title"], r["genre"], r["pub_year"])
            if context
            else body
        )
        ids.append(c["i"])
    return texts, ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--input", choices=("raw", "modern"), default="raw")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-context", action="store_true",
                    help="較正の対照条件: 作品名等の前置きを付けない")
    ap.add_argument("--subset", type=str, default=None,
                    help="対象チャンク番号を書いた JSON(部分実験用)")
    ap.add_argument("--tag", type=str, default=None, help="出力名の上書き")
    a = ap.parse_args()

    import os
    torch.set_num_threads(os.cpu_count() or 4)
    sub = set(json.loads(Path(a.subset).read_text(encoding="utf-8"))) if a.subset else None
    texts, ids = passages(a.input, context=not a.no_context, subset=sub)
    if a.limit:
        texts, ids = texts[: a.limit], ids[: a.limit]
    enc = Encoder(a.model)
    t0 = time.time()
    vecs = []
    step = 512
    for i in range(0, len(texts), step):
        vecs.append(enc.encode(texts[i : i + step], "passage", batch=a.batch))
        done = min(i + step, len(texts))
        el = time.time() - t0
        print(f"  {done}/{len(texts)} ({el:.0f}s, 残り {el/done*(len(texts)-done):.0f}s)", flush=True)
    V = np.concatenate(vecs, axis=0).astype(np.float32)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.model}_{a.input}" + ("_noctx" if a.no_context else "")
    np.save(OUTDIR / f"{tag}.npy", V)
    (OUTDIR / f"{tag}.json").write_text(
        json.dumps({"model": MODELS[a.model]["repo"], "input": a.input,
                    "dim": int(V.shape[1]), "n": int(V.shape[0]), "ids": ids,
                    "built_at": "2026-08-25"}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{V.shape} → {OUTDIR / (tag + '.npy')}  ({time.time()-t0:.0f}s)")
