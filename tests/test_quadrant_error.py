import numpy as np
import pandas as pd
from spatialaudiometrics import localisation_metrics as lm


def make_df(pol_target, pol_response, lat_target, lat_response=None):
    df = pd.DataFrame({'pol_target': pol_target, 'pol_response': pol_response, 'lat_target': lat_target})
    df['lat_response'] = lat_target if lat_response is None else lat_response
    df['polar_weight'] = lm.polar_error_weight(df)
    return df


def test_perfect_and_mirrored():
    df = make_df([0, 45, 170], [0, 45, 170], [0, 0, 0])
    assert lm.calculate_quadrant_error(df, weighted=True)[0] == 0
    df = make_df([0, 45, 170], [180, 225, -10], [0, 0, 0])   # all front-back flipped
    assert lm.calculate_quadrant_error(df, weighted=True)[0] == 100


def test_wrapping():
    # 350 vs 10 deg is a 20 deg error, not 340
    assert lm.calculate_quadrant_error(make_df([350], [10], [0]), weighted=True)[0] == 0


def test_weighted():
    # one frontal trial with a quadrant error (w = 1), one lateral trial correct (w = 0.5*cos(120 deg)+0.5 = 0.25)
    df = make_df([0, 0], [180, 0], [0, 60])
    assert np.isclose(lm.calculate_quadrant_error(df, weighted=True)[0], 100 * 1 / 1.25)


def test_default_is_middlebrooks():
    # the lateral trial (response at 60 deg lateral) is discarded and there are no weights
    df = make_df([0, 0, 0], [180, 0, 180], [0, 0, 60], lat_response=[0, 0, 60])
    qe, confusions, n = lm.calculate_quadrant_error(df)
    assert (qe, confusions, n) == (50, 1, 2)
