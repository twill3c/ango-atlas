"""言及グラフとコミュニティ抽出(F-22)。

固有名詞の共起で作品どうしを結び、Louvain 法でコミュニティを取る。
**辺は本文実測の共起のみ**。推定で辺を足さない(F-22)。

作品 513 個の網をそのまま描いても読めないので、成果物は
「コミュニティ × その特徴的な固有名詞 × 所属作品」の形にする。
コミュニティの性格は、共起で束ねた結果として**語が語る**。

Louvain は近傍の走査順に依存するので、**node 番号順に固定**して再現性を担保する(N-06)。
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "works_meta.json"
OUT = ROOT / "data" / "mentions.json"

MIN_DF = 2      # 2 作品以上に出る固有名詞だけを使う
MAX_DF_RATIO = 0.25  # 4 分の 1 を超える作品に出る語は共通語として落とす
MIN_EDGE = 0.08      # これ未満の重みの辺は張らない(実測の分布から決める)
TOP_NAMES = 12
# 解析器の誤りで固有名詞になる断片(旧仮名「なかつた」の「なかつ」など)。実測で拾った
ARTIFACTS = {"なかつ", "つた", "なかっ", "ゐる", "つて"}


def proper_nouns() -> dict[str, collections.Counter]:
    """作品ごとの固有名詞の出現数。"""
    import fugashi

    from pipeline import aozora_parser as ap

    tagger = fugashi.Tagger()
    meta = json.loads(META.read_text(encoding="utf-8"))["works"]
    out: dict[str, collections.Counter] = {}
    for r in meta:
        cid = r["card_id"]
        with open(RAW / f"{cid}.txt", encoding="utf-8", newline="") as f:
            body = ap.parse(f.read()).body_text
        c = collections.Counter()
        for w in tagger(body):
            if w.feature.pos2 == "固有名詞" and len(w.surface) >= 2:
                c[w.surface] += 1
        out[cid] = c
    return out


def vectors(pn: dict[str, collections.Counter]) -> tuple[list[str], dict[str, dict[str, float]]]:
    df = collections.Counter()
    for c in pn.values():
        df.update(set(c))
    n = len(pn)
    keep = {t for t, v in df.items() if MIN_DF <= v <= n * MAX_DF_RATIO}
    cards = sorted(pn)
    vecs: dict[str, dict[str, float]] = {}
    for cid in cards:
        v = {t: (1 + math.log(f)) * math.log(n / df[t]) for t, f in pn[cid].items() if t in keep}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs[cid] = {t: x / norm for t, x in v.items()}
    return cards, vecs


def edges(cards: list[str], vecs: dict[str, dict[str, float]]) -> list[tuple[int, int, float]]:
    """共有する固有名詞の重み(コサイン)で辺を張る。転置索引で候補を絞る。"""
    inv: dict[str, list[int]] = collections.defaultdict(list)
    for i, cid in enumerate(cards):
        for t in vecs[cid]:
            inv[t].append(i)
    acc: dict[tuple[int, int], float] = collections.defaultdict(float)
    for t, members in inv.items():
        if len(members) < 2:
            continue
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                a, b = members[a_i], members[b_i]
                acc[(a, b)] += vecs[cards[a]][t] * vecs[cards[b]][t]
    return [(a, b, w) for (a, b), w in acc.items() if w >= MIN_EDGE]


def louvain(n: int, es: list[tuple[int, int, float]], rounds: int = 12) -> list[int]:
    """Louvain 法(modularity 最大化の貪欲な局所移動)。走査順を固定して再現する。"""
    adj: list[dict[int, float]] = [dict() for _ in range(n)]
    m2 = 0.0
    for a, b, w in es:
        adj[a][b] = adj[a].get(b, 0.0) + w
        adj[b][a] = adj[b].get(a, 0.0) + w
        m2 += 2 * w
    if m2 == 0:
        return list(range(n))
    k = [sum(d.values()) for d in adj]
    comm = list(range(n))
    tot = k[:]  # コミュニティの次数和
    for _ in range(rounds):
        moved = False
        for v in range(n):  # 走査順は node 番号順に固定(N-06)
            cur = comm[v]
            tot[cur] -= k[v]
            weights: dict[int, float] = collections.defaultdict(float)
            for u, w in adj[v].items():
                weights[comm[u]] += w
            best, best_gain = cur, weights.get(cur, 0.0) - tot[cur] * k[v] / m2
            for c, w in sorted(weights.items()):
                gain = w - tot[c] * k[v] / m2
                if gain > best_gain + 1e-12:
                    best, best_gain = c, gain
            comm[v] = best
            tot[best] += k[v]
            if best != cur:
                moved = True
        if not moved:
            break
    # 番号を詰める
    order = {c: i for i, c in enumerate(sorted(set(comm)))}
    return [order[c] for c in comm]


def build() -> dict:
    pn = proper_nouns()
    cards, vecs = vectors(pn)
    global_df = collections.Counter()
    for c in pn.values():
        global_df.update(set(c))
    es = edges(cards, vecs)
    comm = louvain(len(cards), es)
    meta = {r["card_id"]: r for r in json.loads(META.read_text(encoding="utf-8"))["works"]}

    groups: dict[int, list[str]] = collections.defaultdict(list)
    for cid, c in zip(cards, comm):
        groups[c].append(cid)
    out = []
    for c, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(members) < 3:
            continue
        # そのコミュニティ**らしい**名前を選ぶ。全体で普遍的な語(日本・東京)ではなく、
        # 内側の出現率が全体の出現率より突出しているものを上に置く
        inside = collections.Counter()
        for cid in members:
            inside.update(set(pn[cid]))
        scored = {
            t: (inside[t] / len(members)) * math.log(len(cards) / global_df[t])
            for t in inside
            if inside[t] >= 2 and t not in ARTIFACTS
        }
        picked = [t for t, _ in sorted(scored.items(), key=lambda x: -x[1])[:TOP_NAMES]]
        out.append({
            "id": int(c),
            "size": len(members),
            "names": picked,
            "works": sorted(
                members, key=lambda x: -(meta[x]["chars"] or 0)
            )[:12],
            "genres": dict(collections.Counter(meta[x]["genre"] or "不明" for x in members)),
            "years": [meta[x]["pub_year"] for x in members if meta[x]["pub_year"]],
        })
    return {
        "provenance": {
            "built_at": "2026-08-25",
            "method": "固有名詞 TF-IDF のコサインで作品間の辺を張り、Louvain でコミュニティ",
            "min_df": MIN_DF, "max_df_ratio": MAX_DF_RATIO, "min_edge": MIN_EDGE,
            "note": "辺は本文実測の共起のみ。推定辺は無い",
        },
        "n_works": len(cards),
        "n_edges": len(es),
        "n_communities": len(out),
        "isolated": sum(1 for c in collections.Counter(comm).values() if c < 3),
        "communities": out,
    }


if __name__ == "__main__":
    d = build()
    OUT.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"作品 {d['n_works']} / 辺 {d['n_edges']} / コミュニティ {d['n_communities']}")
    meta = {r["card_id"]: r for r in json.loads(META.read_text(encoding="utf-8"))["works"]}
    for c in d["communities"][:10]:
        print(f"  {c['size']:3d} 作品  {' / '.join(c['names'][:8])}")
        print(f"        代表: {meta[c['works'][0]]['title'][:30]}")
