"""相互リンクと固定フッタ(F-23 / F-24)。

ブラウザを開けないので、静的な検査と node での実データ実行で守る。
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
PAGES = ["index.html", "lens.html", "topic.html", "search.html", "reader.html"]


@pytest.mark.unit
def test_t701_every_page_loads_the_shared_chrome():
    """T-701 / F-23・F-24: 全ページが共通のヘッダ・フッタを読み込む。"""
    for p in PAGES:
        src = (WEB / p).read_text(encoding="utf-8")
        assert '<script src="nav.js"></script>' in src, f"{p} が nav.js を読んでいない"
        assert 'data-page=' in src, f"{p} に data-page が無い"
        assert '<link rel="stylesheet" href="style.css">' in src


@pytest.mark.unit
def test_t702_footer_has_the_five_links():
    """T-702 / F-24: フッタは © / GitHub / 歩き方 / 設計図 / App Menu の 5 つ。

    kiko-atlas 以降のフリート共通の並び。歩き方・設計図は Artifact の URL。
    """
    nav = (WEB / "nav.js").read_text(encoding="utf-8")
    assert "&copy; 2026 twill3c" in nav
    assert "github.com/twill3c/ango-atlas" in nav
    assert "app-menu-amber.vercel.app" in nav
    for anchor in ("link-howto", "link-design"):
        m = re.search(r'id="' + anchor + r'" href="(https://claude\.ai/code/artifact/[0-9a-f-]{36})"', nav)
        assert m, f"{anchor} の Artifact URL が無い"


@pytest.mark.unit
def test_t703_fixed_footer_does_not_hide_the_text():
    """T-703 / F-24: フッタは固定配置で、本文の末尾を隠さない。

    body に下の余白が取ってあることを CSS で確かめる。
    """
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert "position: fixed" in css
    m = re.search(r"body\s*\{[^}]*padding:\s*0\s+1rem\s+(\d+(?:\.\d+)?)rem", css)
    assert m, "body の下余白が読み取れない"
    assert float(m.group(1)) >= 3.0, f"下余白 {m.group(1)}rem ではフッタに隠れる"


@pytest.mark.unit
def test_t704_pages_link_to_each_other_and_to_the_reader():
    """T-704 / F-23: 各ビューからリーダーへ渡れる。ナビは全ページを指す。"""
    nav = (WEB / "nav.js").read_text(encoding="utf-8")
    for target in ("index.html", "lens.html", "topic.html", "search.html", "reader.html"):
        assert target in nav, f"ナビに {target} が無い"
    for p in ("index.html", "lens.html", "topic.html", "search.html"):
        src = (WEB / p).read_text(encoding="utf-8")
        assert "reader.html?w=" in src, f"{p} からリーダーへのリンクが無い"


@pytest.mark.unit
def test_t705_no_dangling_local_links():
    """T-705 / F-23: ページ内のローカルリンク先が実在する。"""
    missing = []
    for p in PAGES:
        src = (WEB / p).read_text(encoding="utf-8")
        for href in re.findall(r'href="([^"#?:]+\.(?:html|css|js))"', src):
            if not (WEB / href).exists():
                missing.append((p, href))
        for src_attr in re.findall(r'<script src="([^"]+)"', src):
            if not (WEB / src_attr).exists():
                missing.append((p, src_attr))
    assert not missing, f"リンク先が無い: {missing}"


@pytest.mark.validation
def test_t706_all_pages_render_with_real_data():
    """T-706 / F-23: 全 5 ページが実データで例外なく描画する(node の DOM スタブ)。"""
    if not shutil.which("node"):
        pytest.skip("node が無い")
    r = subprocess.run(
        ["node", str(ROOT / "tests" / "smoke_pages.js")],
        capture_output=True, text=True, cwd=ROOT, encoding="utf-8",
    )
    assert r.returncode == 0, r.stdout + r.stderr
    for p in PAGES:
        assert f"OK {p}" in r.stdout, r.stdout
