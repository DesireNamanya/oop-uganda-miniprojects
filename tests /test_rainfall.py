"""Unit tests for src/rainfall.py (Mini-Project 4).

Run with:  pytest -q
"""

import math

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine_distance

from src.rainfall import (
    MONTHS,
    CropRule,
    RainfallRecord,
    Region,
    SuitabilityAnalyser,
    cosine_similarity,
    default_crop_rules,
    euclidean_distance,
    flawed_cosine_similarity,
    pairwise_matrix,
    pearson_correlation,
)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


@pytest.fixture
def kampala() -> Region:
    return Region("Kampala", KAMPALA)


# --- Region -----------------------------------------------------------------
def test_region_statistics(kampala):
    assert kampala.annual_total() == 1600
    assert kampala.mean() == pytest.approx(1600 / 12)
    assert kampala.wettest_month() == "May" and kampala.driest_month() == "Sep"
    assert kampala.cv() == pytest.approx(np.std(KAMPALA, ddof=1) / np.mean(KAMPALA))
    assert len(kampala) == 12 and kampala["may"] == 220 and kampala[0] == 120


@pytest.mark.parametrize("bad", [[1] * 11, [1] * 11 + [-5], [1] * 11 + [float("nan")], []])
def test_region_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        Region("X", bad)


def test_all_zero_rainfall_edge_case():
    dry = Region("Desert", [0] * 12)
    assert dry.annual_total() == 0
    assert math.isnan(dry.cv())
    assert dry.modality() == "no clear season"


def test_modality_on_brief_data():
    assert Region("Mbarara", MBARARA).modality() == "bimodal"
    assert Region("Gulu", GULU).modality() == "unimodal"


def test_circular_peak_in_december_is_found():
    # A single peak in December is only visible if the year wraps around
    x = [50, 40, 30, 20, 10, 10, 10, 10, 20, 40, 80, 150]
    seasons = Region("Wrap", x).rainy_seasons()
    assert [s.month for s in seasons] == ["Dec"]


def test_threshold_controls_modality():
    gulu = Region("Gulu", GULU)
    assert gulu.modality(min_relative_prominence=0.1) == "bimodal"   # May shoulder counted
    assert gulu.modality(min_relative_prominence=0.3) == "unimodal"
    with pytest.raises(ValueError):
        gulu.rainy_seasons(1.5)


# --- CropRule ---------------------------------------------------------------
def test_crop_rule_boundaries():
    rule = CropRule("maize", 85, 190)
    assert rule.classify(84.9) == CropRule.DROUGHT
    assert rule.classify(85) == CropRule.GOOD and rule.classify(190) == CropRule.GOOD
    assert rule.classify(190.1) == CropRule.WATERLOG
    assert rule.classify(0) == CropRule.DROUGHT
    assert [rule.score(v) for v in (0, 100, 300)] == [-1, 0, 1]


def test_crop_rule_from_seasonal_need_hand_calculation():
    # maize 500-800 mm over 125-180 days: 500/(180/30)=83.3->85, 800/(125/30)=192->190
    rule = CropRule.from_seasonal_need("maize", (500, 800), (125, 180))
    assert (rule.min_mm, rule.max_mm) == (85, 190)


@pytest.mark.parametrize("args", [("maize", 100, 50), ("maize", -1, 50), ("", 10, 20)])
def test_invalid_crop_rules(args):
    with pytest.raises(ValueError):
        CropRule(*args)
    with pytest.raises(ValueError):
        CropRule("beans", 10, 20).classify(-3)


def test_analyser_labels(kampala):
    analyser = SuitabilityAnalyser([kampala, Region("Gulu", GULU)], default_crop_rules())
    labels = analyser.labels()
    assert labels["Kampala"][4] == "Waterlogging risk"     # May 220 mm > every upper bound
    assert labels["Gulu"][0] == "Drought risk"             # Jan 8 mm
    assert labels["Kampala"][0].startswith("Good for maize")
    grid, rows = analyser.score_grid()
    assert grid.shape == (6, 12) and rows[0] == "Kampala - maize"
    with pytest.raises(ValueError):
        SuitabilityAnalyser([], default_crop_rules())


# --- similarity -------------------------------------------------------------
def test_cosine_matches_scipy():
    for a, b in [(KAMPALA, GULU), (KAMPALA, MBARARA), (GULU, MBARARA)]:
        assert cosine_similarity(a, b) == pytest.approx(1 - scipy_cosine_distance(a, b))


def test_cosine_is_scale_invariant_but_euclidean_is_not():
    a = np.array(KAMPALA, dtype=float)
    assert cosine_similarity(a, 3 * a) == pytest.approx(1.0)
    assert euclidean_distance(a, 3 * a) == pytest.approx(2 * np.linalg.norm(a))


def test_cosine_edge_cases():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cosine_similarity([0, 0, 0], [1, 2, 3])
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_flawed_method_is_not_a_similarity():
    # identical vectors must score 1; math.cos(dot product) does not
    assert flawed_cosine_similarity(KAMPALA, KAMPALA) != pytest.approx(1.0)


def test_pearson_matches_numpy_and_matrix_is_symmetric():
    assert pearson_correlation(KAMPALA, GULU) == pytest.approx(np.corrcoef(KAMPALA, GULU)[0, 1])
    regions = [Region("K", KAMPALA), Region("G", GULU), Region("M", MBARARA)]
    m = pairwise_matrix(regions, cosine_similarity)
    np.testing.assert_allclose(m, m.T)
    np.testing.assert_allclose(np.diag(m), 1.0)


# --- multi-year record --------------------------------------------------------
def test_rainfall_record_and_csv(tmp_path):
    path = tmp_path / "r.csv"
    lines = ["region,year,month,rainfall_mm"]
    for year in (2020, 2021):
        for m in range(1, 13):
            lines.append(f"Test,{year},{m},{m * (1 if year == 2020 else 3)}")
    path.write_text("\n".join(lines))
    rec = RainfallRecord.from_csv(path)["Test"]
    assert len(rec) == 2 and rec.years == [2020, 2021]
    np.testing.assert_allclose(rec.climatology().rainfall, 2 * np.arange(1, 13))
    np.testing.assert_allclose(rec.annual_totals(), [78, 234])
    np.testing.assert_allclose(rec.probability_at_least(10), [0] * 3 + [0.5] * 6 + [1] * 3)


def test_rainfall_record_rejects_missing_value(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("region,year,month,rainfall_mm\nTest,2020,1,-999\n")
    with pytest.raises(ValueError):
        RainfallRecord.from_csv(path)
