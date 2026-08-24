"""文体特徴量の抽出(F-08)。

**定義はテスト側(T-301..305)に手計算フィクスチャとして明文化してある。**
実装を変えるときは、まず手計算の期待値が正しいかを確認すること。

品詞・機能語の特徴は原文ではなく `kana_fold.fold` を掛けた表現の上で取る。
実測(2026-08-25、二重版 12 組 × 無関係 30 組)では、畳むと作品間の判別力はほぼ
変わらないまま(L1 0.1099 → 0.1057)、版間の差だけが 9 分の 1 になった
(L1 0.0218 → 0.0023、S/N 5.0 → 45.8)。詳細は docs/style_features_calibration.md。

短編と長編で分散が変わるため、長い作品は一定長のチャンクに切って**チャンクごとの
特徴量の平均**を作品の値とする(F-08 の層別要求)。
"""
from __future__ import annotations

import re
import statistics
from functools import lru_cache

from pipeline import kana_fold as kf

CHUNK = 4000  # 字。これ以上の作品はチャンクに切って平均を取る

_KANJI = re.compile(r"[一-鿿々]")
_HIRA = re.compile(r"[ぁ-ゖゝゞ]")
_KATA = re.compile(r"[ァ-ヴーヽヾ]")
_SENT_END = "。！？"
_INDENT = "　 \t"

# 一人称。表記ゆれは fold 後の形も見る(わたくし → わたくし のまま)
FIRST_PERSON = ("私", "僕", "俺", "わたし", "あたし", "自分", "余", "小生")
# 文末表現(文の末尾に現れる形)
SENT_END_FORMS = ("である", "だ", "です", "ます", "だろう", "であろう", "らしい", "のだ")


@lru_cache(maxsize=4)
def _tagger():
    import fugashi

    return fugashi.Tagger()


def _plain(text: str) -> str:
    """行頭の字下げと改行を落とした本文字列。長さの分母になる。"""
    return "".join(line.strip(_INDENT) for line in text.splitlines())


def sentences(text: str) -> list[str]:
    """句点・感嘆符・疑問符で文に切る。行頭の字下げは長さに数えない。"""
    plain = _plain(text)
    out, buf = [], []
    for ch in plain:
        buf.append(ch)
        if ch in _SENT_END:
            out.append("".join(buf))
            buf = []
    if buf and "".join(buf).strip():
        out.append("".join(buf))
    return out


def speech_chars(text: str) -> int:
    """鉤括弧の中身の字数。閉じ括弧が無い場合は行末までとみなす。"""
    total = 0
    for line in text.splitlines():
        depth = 0
        start = 0
        for i, ch in enumerate(line):
            if ch in "「『":
                if depth == 0:
                    start = i + 1
                depth += 1
            elif ch in "」』":
                if depth == 1:
                    total += i - start
                depth = max(0, depth - 1)
        if depth > 0:
            total += len(line) - start
    return total


def _chunk_features(text: str, fold_pos: bool = True) -> dict[str, float]:
    plain = _plain(text)
    n = len(plain)
    if n == 0:
        return {}
    ss = sentences(text)
    lens = [len(s) for s in ss] or [0]
    kuten = plain.count("。") + plain.count("！") + plain.count("？")
    touten = plain.count("、")
    f = {
        "n_chars": float(n),
        "sent_len_mean": statistics.fmean(lens),
        "sent_len_var": statistics.pvariance(lens) if len(lens) > 1 else 0.0,
        "kanji_ratio": len(_KANJI.findall(plain)) / n,
        "hiragana_ratio": len(_HIRA.findall(plain)) / n,
        "katakana_ratio": len(_KATA.findall(plain)) / n,
        "kuten_per_1000": kuten / n * 1000,
        "touten_per_1000": touten / n * 1000,
        "punct_per_1000": (kuten + touten) / n * 1000,
        "speech_ratio": speech_chars(text) / n,
    }

    # fold_pos=False は較正用の対照条件(原文のまま品詞を取る)
    folded = kf.fold(plain) if fold_pos else plain
    words = list(_tagger()(folded))
    total = len(words) or 1
    pos = {}
    func = {}
    for w in words:
        pos[w.feature.pos1] = pos.get(w.feature.pos1, 0) + 1
        if w.feature.pos1 in ("助詞", "助動詞"):
            func[w.surface] = func.get(w.surface, 0) + 1
    for k in ("名詞", "動詞", "形容詞", "副詞", "助詞", "助動詞", "接続詞", "代名詞", "感動詞"):
        f[f"pos_{k}"] = pos.get(k, 0) / total
    for k in FUNC_WORDS:
        f[f"func_{k}"] = func.get(k, 0) / total
    # 一人称・文末表現は畳んだ本文の上で数える(表記ゆれを吸収するため)
    for p in FIRST_PERSON:
        f[f"fp_{p}"] = folded.count(kf.fold(p)) / n * 1000
    for e in SENT_END_FORMS:
        fe = kf.fold(e)
        f[f"end_{e}"] = sum(1 for s in ss if kf.fold(s).rstrip("。！？").endswith(fe)) / len(ss)
    return f


# 機能語ベクトルの語彙。**実測で決める**(記憶で書くと、UniDic が動詞と判定する
# 「ある・いる」等を入れてしまい全件 0 の退化特徴量になる — HC-022 の分布検査で検出)。
# 出所: コーパスから無作為 80 作品 × 各 12,000 字を fold して助詞・助動詞の表層形を
# 数えた上位 30(2026-08-25 実測、総出現 134,597 / 異なり 259)。
FUNC_WORDS = (
    "の", "に", "て", "わ", "が", "で", "と", "た", "も", "な", "だ", "から", "か",
    "ない", "つ", "れ", "や", "ば", "ん", "です", "ず", "よ", "だけ", "まで", "ぬ",
    "られ", "など", "ね", "たり", "なら",
)


def features(text: str, chunk: int = CHUNK, fold_pos: bool = True) -> dict[str, float]:
    """作品の特徴量。長い作品はチャンクに切って平均を取る(層別)。"""
    plain_len = len(_plain(text))
    if plain_len <= chunk:
        return _chunk_features(text, fold_pos)
    # 段落境界を保ったままチャンクに切る
    parts, buf, size = [], [], 0
    for line in text.splitlines(keepends=True):
        buf.append(line)
        size += len(line.strip(_INDENT + chr(10) + chr(13)))
        if size >= chunk:
            parts.append("".join(buf))
            buf, size = [], 0
    if buf:
        tail = "".join(buf)
        # 末尾の極小チャンクは直前へ併合する。等重みで平均すると数十字のかけらが
        # 作品全体の比率を歪める(実測: card45744 で平仮名率 0.607 → 0.406)
        if parts and len(_plain(tail)) < chunk // 2:
            parts[-1] += tail
        else:
            parts.append(tail)
    feats = [_chunk_features(p, fold_pos) for p in parts]
    feats = [f for f in feats if f]
    keys = feats[0].keys()
    out = {k: statistics.fmean(f[k] for f in feats) for k in keys}
    out["n_chars"] = float(plain_len)
    out["n_chunks"] = float(len(feats))
    return out
