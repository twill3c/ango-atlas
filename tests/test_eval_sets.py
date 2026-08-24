"""評価オラクルそのものの検査(F-14)。

オラクルが壊れていれば、その上で測った Recall はすべて無意味になる。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    p = ROOT / "data" / name
    if not p.exists():
        pytest.skip(f"{name} 未生成")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.validation
def test_t412_eval_queries_point_at_real_chunks():
    """T-412 / F-14: 人手評価セットの正解チャンクが実在し、想定作品に属する。"""
    q = _load("eval_queries.json")
    ch = {c["i"]: c for c in _load("chunks.json")["chunks"]}
    meta = {r["card_id"]: r for r in _load("works_meta.json")["works"]}
    assert len(q["queries"]) >= 30, "F-14 は 30〜50 問を要求している"
    for e in q["queries"]:
        assert e["gold"], e
        assert e["q"] and e["why"], e
        for g in e["gold"]:
            assert g in ch, f"存在しないチャンク {g}"
            title = meta[ch[g]["card_id"]]["title"]
            assert title.startswith(e["work"][:6]) or e["work"] in title, (
                f"{g} は {title} に属し、想定の {e['work']} と違う"
            )


@pytest.mark.validation
def test_t413_eval_queries_are_not_lexical_giveaways():
    """T-413 / F-14: 問いが正解チャンクの珍しい語をそのまま含んでいない。

    語の一致だけで解けてしまう問いは、意味検索の評価にならない。ただし
    「〜ようになった」のような機能語の連なりは避けようがなく、害も無い。
    そこで基準を **6 文字以上の共通部分列に漢字が 2 つ以上含まれないこと** とする
    (内容語がそのまま写っているかどうかを見る)。
    """
    import re

    kanji = re.compile(r"[一-鿿々]")
    q = _load("eval_queries.json")
    ch = {c["i"]: c for c in _load("chunks.json")["chunks"]}
    bad = []
    for e in q["queries"]:
        text = "".join(ch[g]["text"] for g in e["gold"] if g in ch)
        for n in range(len(e["q"]), 5, -1):
            hit = next((e["q"][i : i + n] for i in range(len(e["q"]) - n + 1)
                        if e["q"][i : i + n] in text), None)
            if hit:
                if len(kanji.findall(hit)) >= 2:
                    bad.append((e["q"][:26], hit))
                break
    assert not bad, f"問いに正解本文の内容語がそのまま入っている: {bad[:5]}"


@pytest.mark.validation
def test_t414_variant_chunk_oracle_is_clean():
    """T-414 / F-14: 二重版チャンク対応が一対一で、別作品を指していない。"""
    ev = _load("eval_variant_chunks.json")
    ch = {c["i"]: c for c in _load("chunks.json")["chunks"]}
    seen_q, seen_g = set(), set()
    for e in ev["pairs"]:
        assert e["query_chunk"] not in seen_q, "同じ問いが 2 回ある"
        assert e["gold_chunk"] not in seen_g, "同じ正解が 2 回使われている"
        seen_q.add(e["query_chunk"])
        seen_g.add(e["gold_chunk"])
        assert ch[e["query_chunk"]]["card_id"] == e["old_card"]
        assert ch[e["gold_chunk"]]["card_id"] == e["new_card"]
        assert e["score"] >= ev["provenance"]["min_score"]
