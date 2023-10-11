"""Tests for tobac_flow.detection module"""

import numpy as np
from tobac_flow.flow import Flow


def test_get_combined_edge_field():
    from tobac_flow.detection import get_combined_edge_field

    test_field = np.zeros([1, 5, 5], dtype=np.float32)
    test_field[:, 3:] = 1

    test_flow = Flow(
        np.zeros([1, 5, 5, 2], dtype=np.float32),
        np.zeros([1, 5, 5, 2], dtype=np.float32),
    )

    test_result = get_combined_edge_field(test_flow, test_field)

    # Check basic results: Positive valued at edge, zero where field is zero and negative where field is 1
    assert np.all(test_result[:, 2] > 0)
    assert np.all(test_result[:, :2] == 0)
    assert np.all(test_result[:, 3:] == -1)

    # Test missing values

    test_field[:, :, 0] = np.nan

    test_result = get_combined_edge_field(test_flow, test_field)

    assert np.all(np.isnan(test_field) == np.isinf(test_result))
