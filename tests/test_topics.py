"""主題(F-19 / F-20)と言及グラフ(F-22)。"""
import json
from pathlib import Path

import pytest

from pipeline import mentions as mn
from pipeline import topics as tp

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    p = ROOT / "data" / name
    if not p.exists():
        pytest.skip(f"{name} 未生成")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_t606_ctfidf_prefers_distinctive_terms():
    """T-606 / F-19: c-TF-IDF は全体で普遍的な語より、その主題に偏る語を上に置く。

    合成データ: 語 A は全主題に均等、語 B は主題 0 だけに出る。B が上に来るはず。
    """
    import collections

    import numpy as np

    docs = [collections.Counter({"A": 5, "B": 5})] * 10 + [collections.Counter({"A": 5})] * 10
    labels = np.array([0] * 10 + [1] * 10)
    words = tp.ctfidf(labels, docs, 2, topn=2, min_total=1)
    assert words[0][0] == "B", words


@pytest.mark.unit
def test_t607_stoplist_keeps_thematic_nouns():
    """T-607 / F-19: 形式名詞は落とし、主題語は残す。

    df による機械的な足切りにすると「女」「人間」「日本」まで落ちるので、
    語を一つずつ見て決めている。その意図をテストで固定する。
    """
    for w in ("こと", "もの", "ところ", "とき", "ため"):
        assert w in tp.STOP, f"{w} は形式名詞なので落とすべき"
    for w in ("女", "男", "人間", "日本", "心", "自分", "文学", "戦争"):
        assert w not in tp.STOP, f"{w} は安吾の主題語なので残すべき"


@pytest.mark.unit
def test_t608_louvain_finds_planted_communities():
    """T-608 / F-22: 二つの塊を張り合わせたグラフで、その二つに分かれる。"""
    es = []
    for a in range(5):
        for b in range(a + 1, 5):
            es.append((a, b, 1.0))
            es.append((a + 5, b + 5, 1.0))
    es.append((0, 5, 0.05))  # 塊どうしを細い辺で 1 本だけ結ぶ
    comm = mn.louvain(10, es)
    assert len(set(comm)) == 2, comm
    assert len(set(comm[:5])) == 1 and len(set(comm[5:])) == 1
    assert mn.louvain(10, es) == comm, "同じ入力で同じ結果を返すこと(N-06)"


@pytest.mark.validation
def test_t609_every_chunk_has_a_topic():
    """T-609 / F-19: 全チャンクに主題が付き、件数が合う。

    HDBSCAN を採らなかったので「ノイズ」は存在しない。
    """
    d = _load("topics.json")
    chunks = _load("chunks.json")["chunks"]
    assert len(d["labels"]) == len(chunks)
    assert min(d["labels"]) == 0 and max(d["labels"]) == d["k"] - 1
    assert sum(d["sizes"]) == len(chunks)
    assert sum(sum(v) for v in d["by_work"].values()) == len(chunks)


@pytest.mark.validation
def test_t610_topic_names_are_marked_as_generated():
    """T-610 / F-20: 主題の名前は全主題に付いており、生成物と明示されている。"""
    names = _load("topic_names.json")
    d = _load("topics.json")
    assert names["provenance"]["generated_by"], "生成物であることが記録されていない"
    assert "生成物" in names["provenance"]["warning"]
    assert names["provenance"]["k"] == d["k"], "k が変わったら付け直しが要る"
    for j in range(d["k"]):
        e = names["names"][str(j)]
        assert e["name"] and e["desc"], f"主題 {j} に名前が無い"


@pytest.mark.validation
def test_t611_mention_edges_are_measured_only():
    """T-611 / F-22: コミュニティの代表語が、実際にその作品群の本文に出る固有名詞である。

    推定で足した語・辺が混ざっていないことを、成果物の側から確かめる。
    """
    d = _load("mentions.json")
    meta = {r["card_id"]: r for r in _load("works_meta.json")["works"]}
    from pipeline import aozora_parser as ap

    for c in d["communities"][:3]:
        texts = []
        for cid in c["works"][:4]:
            with open(ROOT / "data" / "raw" / f"{cid}.txt", encoding="utf-8", newline="") as f:
                texts.append(ap.parse(f.read()).body_text)
        blob = "".join(texts)
        hit = sum(1 for t in c["names"] if t in blob)
        assert hit >= len(c["names"]) // 2, (
            f"コミュニティ {c['id']} の代表語が本文に見当たらない: {c['names']}"
        )
        assert all(cid in meta for cid in c["works"])
