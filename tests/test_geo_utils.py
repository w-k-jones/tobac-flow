"""
Test utils.geo_utils
"""
import pytest
import numpy as np
from tobac_flow.utils import geo_utils


def test_get_mean_object_azimuth_and_speed() -> None:
    direction, speed = geo_utils.get_mean_object_azimuth_and_speed(
        np.array([0, 0]), np.array([0, 1]), np.array([0, 100e9])
    )
    assert direction == pytest.approx(0)
    assert speed == pytest.approx(1100, 20)

    direction, speed = geo_utils.get_mean_object_azimuth_and_speed(
        np.array([0, 0]), np.array([0, -1]), np.array([0, 100e9])
    )
    assert direction == pytest.approx(180)
    assert speed == pytest.approx(1100, 20)

    direction, speed = geo_utils.get_mean_object_azimuth_and_speed(
        np.array([0, 1]), np.array([0, 0]), np.array([0, 100e9])
    )
    assert direction == pytest.approx(90)
    assert speed == pytest.approx(1100, 20)

    direction, speed = geo_utils.get_mean_object_azimuth_and_speed(
        np.array([0, -1]), np.array([0, 0]), np.array([0, 100e9])
    )
    assert direction == pytest.approx(-90)
    assert speed == pytest.approx(1100, 20)

    direction, speed = geo_utils.get_mean_object_azimuth_and_speed(
        np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([0, 100e9, 150e9])
    )
    assert direction == pytest.approx(45, 0.1)
    assert speed == pytest.approx(1650, 30)
