import numpy as np
import pytest
import pandas as pd
from spatialaudiometrics import localisation_metrics as lm


def make_df(pol_target, pol_response, lat_target, lat_response=None):
    df = pd.DataFrame({'pol_target': pol_target, 'pol_response': pol_response, 'lat_target': lat_target})
    df['lat_response'] = lat_target if lat_response is None else lat_response
    df['polar_weight'], _ = lm.polar_error_weight(df)
    return df


def test_perfect_and_mirrored():
    df = make_df([0, 45, 170], [0, 45, 170], [0, 0, 0])
    assert lm.calculate_quadrant_error(df)[0] == 0
    df = make_df([0, 45, 170], [180, 225, -10], [0, 0, 0])   # all front-back flipped
    assert lm.calculate_quadrant_error(df)[0] == 100


def test_wrapping():
    # 350 vs 10 deg is a 20 deg error, not 340
    assert lm.calculate_quadrant_error(make_df([350], [10], [0]))[0] == 0


def test_weighted():
    # one frontal trial with a quadrant error (w = 1), one lateral trial correct (w = cos(60 deg) = 0.5)
    df = make_df([0, 0], [180, 0], [0, 60])
    assert np.isclose(lm.calculate_quadrant_error(df)[0], 100 * 1 / 1.5)


def test_split_is_middlebrooks():
    # the lateral trial (target at 60 deg lateral) is discarded and there are no weights
    df = make_df([0, 0, 0], [180, 0, 180], [0, 0, 60])
    qe, confusions, n = lm.calculate_quadrant_error(df, weighted_method='split')
    assert (qe, confusions, n) == (50, 1, 2)


def test_split_weight():
    # constant weight for the targets inside 30 deg, the others are not used; the response does not matter
    df = make_df([0, 0, 0, 0], [0, 0, 0, 0], [0, 60, 30, 31], lat_response=[60, 0, 0, 0])
    w, used = lm.polar_error_weight(df, method='split')
    assert list(w) == [1, 1, 1, 1] and list(used) == [True, False, True, False]


def test_localisation_error_weighted_method():
    # two trials with a 10 deg polar error: frontal (w = 1) and lateral (60 deg, w = cos(60 deg) = 0.5)
    df = make_df([0, 0], [10, 10], [0, 60])
    df['g'] = 1
    df['confusion_classification'] = 'precision'
    f = lambda m: lm.calculate_localisation_error(df, 'g', weighted_method=m).pol_local_abs_accuracy[0]
    assert np.isclose(f('none'), 10)
    assert np.isclose(f('cos'), 7.5)
    assert np.isclose(f('split'), 10)   # the lateral response is left out, not counted as zero error


def test_polar_local_metrics():
    # polar errors of 10, -20 and a front-back confusion (170): the confusion is only left out of the local metrics (before weighting)
    df = make_df([0, 0, 0], [10, -20, 170], [0, 0, 0])
    df['g'] = 1
    df['confusion_classification'] = 'precision'
    out = lm.calculate_localisation_error(df, 'g')
    assert np.isclose(out.pol_local_abs_accuracy[0], 15)
    assert np.isclose(out.pol_local_accuracy[0], -5)
    assert np.isclose(out.quadrant_error[0], 100 / 3)
    assert not any(c.startswith('pol_') and not c.startswith('pol_local') for c in out.columns)
    # the global ones include the confusion, with a warning
    with pytest.warns(UserWarning, match='quadrant_error'):
        out = lm.calculate_localisation_error(df, 'g', pol_local=False)
    assert np.isclose(out.pol_abs_accuracy[0], 200 / 3)
    assert not any(c.startswith('pol_local') for c in out.columns)
    # a 350 vs 10 deg response is a 20 deg error, wrapped before the weighting
    df = make_df([350], [10], [60])
    df['g'] = 1
    df['confusion_classification'] = 'precision'
    assert np.isclose(lm.calculate_localisation_error(df, 'g').pol_local_abs_accuracy[0], 20 * 0.5)
