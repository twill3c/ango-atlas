"""文体の転換点検出(F-21)。

**通説との一致は assert しない**(観察として出すだけ — SPEC F-21)。
検証するのは、検出器そのものが正しく動くことと、再現することだけ。
"""
import numpy as np
import pytest

from pipeline import changepoint as cpt


@pytest.mark.unit
def test_t601_pelt_finds_a_planted_shift():
    """T-601 / F-21: 人工的に段差を入れた系列で、その位置を当てる。

    期待値は合成データの作り方から導かれる(段差は 30 と 70 に置いた)。
    導出の前提(段差が雑音より十分大きい)をテスト内で検算する。
    """
    rng = np.random.default_rng(0)
    x = np.concatenate([
        rng.normal(0.0, 0.2, 30), rng.normal(3.0, 0.2, 40), rng.normal(-2.0, 0.2, 30)
    ])
    # 前提の検算: 段差(3.0 / 5.0)は雑音の標準偏差 0.2 より十分大きい
    assert abs(x[:30].mean() - x[30:70].mean()) > 10 * 0.2
    cps = cpt.pelt(x, penalty=cpt.penalty_for(len(x), 1))
    assert cps == [30, 70], cps


@pytest.mark.unit
def test_t602_pelt_finds_nothing_in_noise():
    """T-602 / F-21: 段差の無い系列では変化点を返さない。"""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 120)
    cps = cpt.pelt(x, penalty=cpt.penalty_for(len(x), 1))
    assert cps == [], cps


@pytest.mark.unit
def test_t605_penalty_is_calibrated_not_theoretical():
    """T-605 / F-21: 罰則の倍率は合成データの較正で決まっている。

    理論値そのまま(x1.0)だと純雑音の 8 割で変化点を「発見」する。
    採用値で誤検出が出ないことを、少数の試行で確かめる。
    """
    rng = np.random.default_rng(7)
    fp = sum(
        1 for _ in range(30)
        if cpt.pelt(rng.normal(0, 1, 120), cpt.penalty_for(120, 1))
    )
    assert fp == 0, f"純雑音 30 本で {fp} 本に変化点が出た"
    assert cpt.PENALTY_MULT > 1.0, "理論値そのままでは誤検出が多すぎる"


@pytest.mark.unit
def test_t603_multivariate_and_reproducible():
    """T-603 / F-21・N-06: 多変量でも動き、同じ入力で同じ結果を返す。"""
    rng = np.random.default_rng(2)
    X = np.vstack([
        rng.normal(0.0, 0.2, (40, 3)), rng.normal(2.0, 0.2, (40, 3))
    ])
    a = cpt.pelt(X, penalty=cpt.penalty_for(len(X), 3))
    b = cpt.pelt(X, penalty=cpt.penalty_for(len(X), 3))
    assert a == b == [40]


@pytest.mark.validation
def test_t604_corpus_changepoints_are_recorded_not_asserted():
    """T-604 / F-21: 実データの結果は**記録**する。通説との一致は要求しない。

    成果物に、検出した年・使った特徴量・通説区分との重なりが残っていることだけを見る。
    """
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "changepoints.json"
    if not p.exists():
        pytest.skip("changepoints.json 未生成")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["years"], "年の系列が空"
    assert "changepoints" in d and isinstance(d["changepoints"], list)
    assert d["features"], "使った特徴量が記録されていない"
    assert "received_periods" in d, "通説区分との対照が記録されていない"
    for c in d["changepoints"]:
        assert d["years"][0] <= c["year"] <= d["years"][-1]
