#!/usr/bin/env python3
"""text_hygiene.py — 日本語本文への別字種・制御文字の混入を検出する(HC-072)。

Python 標準ライブラリのみで動作する。`looplog.py` が全プロジェクトで動いているので、
言語を問わずこの一本で足りる(起票時の案は「Python 版と TypeScript/vitest 版」だったが、
二本持つと片方だけ直るため一本に畳んだ)。

**なぜ要るのか。** 日本語の文中に紛れたキリル文字やハングルは、字形が近く目視では
気づけない。制御文字はさらに悪く、**構文エラーにならない壊れ方**をする。
どちらも「読んでも分からない」故障なので、機械に見張らせるしかない。

使い方:
  python harness/text_hygiene.py                 # 既定の走査対象を見る
  python harness/text_hygiene.py --self-test     # 検査器自身の対照だけを走らせる
  python harness/text_hygiene.py path ...        # 対象を明示する

除外:
  - `.text-hygiene-ignore` に 1 行 1 パターン(部分一致)を書くと、その経路を飛ばす
  - 行内に `text-hygiene:allow` があればその行を飛ばす(記述例のため)

**この検査器自身も走査対象である。** 字種の正規表現と自己対照は当然この検査に引っかかるので、
該当行に `text-hygiene:allow` を付けてある。`harness/` を丸ごと除外しないのは、
そこに置く道具のソースにも混入が起きうるため。

終了コード: 違反 0 件で 0、1 件以上で 1。検査器の自己対照に失敗したら 2。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 走査する拡張子。データではなく**人が書いた文**を対象にする
TEXT_SUFFIXES = {
    ".md", ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".css", ".html", ".json", ".yml", ".yaml", ".toml", ".r", ".rb", ".rs", ".sh",
}

# 既定で見ないところ。**実測(2026-08-31・66 プロジェクト)で決めた**。
#
#   生成物・依存        .git / node_modules / .next / out / dist / build / target / …
#   外部データ          data / content / public —— 青空文庫の本文などが入る。
#                       ここのキリル文字は原文であって混入ではない
#   検査器と記述例      tests / docs —— 検査器自身が字種の正規表現を持ち、
#                       文書は故障の実例を本文に書く。**除外を持たない検査は必ず偽陽性を出す**
#
# 絞る前は 16 プロジェクトで 66 件出て、**そのすべてが正当**だった。
SKIP_DIRS = {
    ".git", "node_modules", ".next", "out", "dist", "build", "target",
    ".venv", "venv", "__pycache__", ".vercel", "logs", "data", "content", "public",
    "coverage", ".pytest_cache", "renv", "tests", "test", "docs", "__tests__",
}

CYRILLIC = re.compile(r"[Ѐ-ӿԀ-ԯ]")  # text-hygiene:allow
HANGUL = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")  # text-hygiene:allow
ALLOWED_CONTROL = {9, 10, 13}  # タブ・改行・復帰
ALLOW_MARKER = "text-hygiene:allow"


def has_control(text: str) -> bool:
    return any((ord(c) < 32 or ord(c) == 127) and ord(c) not in ALLOWED_CONTROL for c in text)


def scan_line(line: str) -> list[str]:
    """その 1 行に何が紛れているか。**行に allow 印があれば何も言わない**。"""
    if ALLOW_MARKER in line:
        return []
    bad = []
    if CYRILLIC.search(line):
        bad.append("キリル文字")
    if HANGUL.search(line):
        bad.append("ハングル")
    if has_control(line):
        bad.append("制御文字")
    return bad


def self_test() -> list[str]:
    """検査器自身の陽性・陰性対照。**これが通らないうちは走査に入らない**。

    「違反 0 件」は「検査した」を意味しない。検査器が死んでいても同じ 0 件が出る。
    """
    errs: list[str] = []
    nul = chr(0)
    positives = [
        ("измерение を含む文", "キリル文字"),  # text-hygiene:allow
        ("あれば그 集合だけが", "ハングル"),  # text-hygiene:allow
        (f'const key = tool + "{nul}" + rest;', "制御文字"),
        (f"末尾に{chr(127)}が居る", "制御文字"),
    ]
    for text, want in positives:
        if want not in scan_line(text):
            errs.append(f"陽性対照が捕まらない: {want}")
    negatives = [
        "ふつうの日本語と ASCII と 記号 —— 全角括弧()",
        "タブ\tと改行の手前まで",
        "ギリシャ文字 γ や数式記号 ＋ は対象外",
        "エスケープとして書いた " + chr(92) + "u0000 は本物ではない",
    ]
    for text in negatives:
        got = scan_line(text)
        if got:
            errs.append(f"陰性対照を撃った: {text[:24]} → {got}")
    # allow 印は効くが、印の無い同じ行は捕まる
    if scan_line(f"измерение {ALLOW_MARKER}"):  # text-hygiene:allow
        errs.append("allow 印が効いていない")
    if not scan_line("измерение"):  # text-hygiene:allow
        errs.append("allow 印が広すぎる")
    return errs


def load_ignores(root: Path) -> list[str]:
    f = root / ".text-hygiene-ignore"
    if not f.exists():
        return []
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def walk(root: Path, ignores: list[str]):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
            continue
        if any(pat in rel for pat in ignores):
            continue
        yield p


def main(argv: list[str]) -> int:
    errs = self_test()
    if errs:
        print("検査器の自己対照に失敗:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 2
    if "--self-test" in argv:
        print("自己対照 OK(陽性 4 / 陰性 4 / allow 印 2)")
        return 0

    root = Path.cwd()
    targets = [Path(a) for a in argv if not a.startswith("--")]
    ignores = load_ignores(root)
    files = targets if targets else list(walk(root, ignores))

    hits = 0
    scanned = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        for i, line in enumerate(text.splitlines(), 1):
            bad = scan_line(line)
            if bad:
                hits += 1
                print(f"{f}:{i}: {', '.join(bad)} — {line.strip()[:80]}")

    print(f"走査 {scanned} ファイル / 違反 {hits} 件")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
