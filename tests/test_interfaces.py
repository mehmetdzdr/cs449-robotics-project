import numpy as np

from conveyor_sort.interfaces import BoxColor, Track


def test_track_constant_velocity_prediction():
    track = Track(
        track_id=0,
        color=BoxColor.BLUE,
        position=np.array([0.0, 0.0, 0.7]),
        velocity=np.array([0.1, 0.0, 0.0]),
        timestamp=1.0,
    )
    np.testing.assert_allclose(track.predict(3.0), [0.2, 0.0, 0.7])
