import numpy as np
from tobac_flow import analysis


def test_find_object_lengths():
    # Test that an array with no labels returns a 0 length array
    empty_array = np.zeros([3]).astype(int)
    assert (
        analysis.find_object_lengths(empty_array).size == 0
    ), "Empty array should return 0 length array"

    one_label = np.array([0, 1, 0]).astype(int)
    assert (
        analysis.find_object_lengths(one_label).size == 1
    ), "Single label should return length 1 array"
    assert analysis.find_object_lengths(one_label)[0] == 1, "Label should be length 1"

    l3_labels = np.array([[1, 1, 1]]).astype(int)
    assert (
        analysis.find_object_lengths(l3_labels)[0] == 1
    ), "Label should be length 1 along axis 0"
    assert (
        analysis.find_object_lengths(l3_labels, axis=1)[0] == 3
    ), "Label should be length 3 along axis 1"

    multiple_labels = np.arange(10).astype(int)
    assert (
        analysis.find_object_lengths(multiple_labels).size == 9
    ), "9 labels shoddl return length 9 array"
    assert np.all(
        analysis.find_object_lengths(multiple_labels) == np.ones([9])
    ), "All 9 labels should have length 1"


def test_mask_labels():
    empty_array = np.zeros([3]).astype(int)
    assert (
        analysis.mask_labels(empty_array, empty_array).size == 0
    ), "Empty array should return 0 length array"

    one_label = np.array([0, 1, 0]).astype(int)
    assert (
        analysis.mask_labels(one_label, empty_array).size == 1
    ), "Single label should return length 1 array"
    assert (
        analysis.mask_labels(one_label, empty_array)[0] == False
    ), "Mask should be False"
    assert analysis.mask_labels(one_label, one_label)[0] == True, "Mask should be True"
