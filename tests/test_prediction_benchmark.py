import numpy as np

from conveyor_sort.prediction.benchmark import prediction_errors

HORIZONS = [0.5, 2.0, 5.0]


def test_errors_grow_with_horizon_and_noise_and_kalman_wins_with_a_long_history():
    errors = {(p, noise): prediction_errors(p, noise, HORIZONS, frames=40, trials=100)
              for p in ("cv", "kalman") for noise in (0.0, 0.002, 0.005)}
    for (predictor, noise), e in errors.items():
        assert e.shape == (100, 3)
        mean = e.mean(axis=0)
        if noise == 0.0:
            assert mean.max() < 1e-4                                   # exact without noise
        else:
            assert mean[0] < mean[1] < mean[2]                         # further ahead is harder
    for predictor in ("cv", "kalman"):                                 # more noise, more error
        assert np.all(errors[predictor, 0.002].mean(axis=0) < errors[predictor, 0.005].mean(axis=0))
    for noise in (0.002, 0.005):                                       # 40 frames: Kalman uses all, the fit its last 20
        assert np.all(errors["kalman", noise].mean(axis=0) < 0.8 * errors["cv", noise].mean(axis=0))


def test_same_seed_gives_same_result():
    assert np.array_equal(prediction_errors("cv", 0.002, HORIZONS, trials=5), prediction_errors("cv", 0.002, HORIZONS, trials=5))
