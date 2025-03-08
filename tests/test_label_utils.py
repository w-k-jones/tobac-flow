import numpy as np
from tobac_flow.utils import label_utils


def test_apply_func_to_labels():
    from tobac_flow.utils.label_utils import apply_func_to_labels

    test_labels = np.zeros([4, 6])
    test_labels[1:3, 1:3] = 1
    test_labels[2:3, 3:6] = 3
    test_labels = test_labels.astype(int)
    test_data1 = np.arange(24).reshape([4, 6])
    test_data2 = np.array([1, 2, 3, 3, 2, 1])

    apply_func_to_labels(test_labels, test_data1, func=np.mean)
    apply_func_to_labels(test_labels, np.stack([test_data1, test_data1]), func=np.mean)
    apply_func_to_labels(
        test_labels,
        np.stack([test_data1, test_data1]),
        func=np.mean,
        index=[1, 3],
    )
    apply_func_to_labels(
        test_labels,
        np.stack([test_data1, test_data1]),
        func=np.mean,
        index=[1, 2, 3, 4],
    )
    apply_func_to_labels(
        test_labels,
        np.stack([test_data1, test_data1]),
        func=np.mean,
        index=[0, 1, 2, 3, 4],
    )
    apply_func_to_labels(
        test_labels,
        np.stack([test_data1, test_data1]),
        func=np.mean,
        index=[-1, 1, 2, 3, 4],
    )

    weighted_mean = lambda a, w: np.average(a, weights=w)
    apply_func_to_labels(test_labels, test_data1, 1, func=weighted_mean)
    apply_func_to_labels(test_labels, test_data1, test_data2, func=weighted_mean)
    apply_func_to_labels(
        test_labels, np.stack([test_data1, test_data1]), 1, func=weighted_mean
    )

    mean_and_std = lambda a: (np.mean(a), np.std(a))
    apply_func_to_labels(test_labels, test_data1, func=mean_and_std)

    weighted_mean_and_std = lambda a, w: (np.average(a, weights=w), np.std(a))
    apply_func_to_labels(
        test_labels, test_data1, test_data2, func=weighted_mean_and_std
    )

    apply_func_to_labels(
        test_labels, test_data1, test_data2, func=weighted_mean_and_std, default=np.nan
    )

    apply_func_to_labels(
        test_labels,
        test_data1,
        test_data2,
        func=weighted_mean_and_std,
        default=[np.nan] * 2,
    )

    apply_func_to_labels(
        test_labels, test_data1, test_data2, func=weighted_mean, default="nan"
    )

    apply_func_to_labels(
        test_labels, test_data1, test_data2, func=weighted_mean, default=["nan"]
    )


def test_slice_labels():
    from tobac_flow.utils.label_utils import slice_labels

    test_labels = np.zeros([5, 10, 15], dtype=np.int32)

    # Add one label
    test_labels[:, 3:6, 4:8] = 1
    sliced_labels = slice_labels(test_labels)
    assert np.all(np.unique(sliced_labels) == np.arange(6))

    # Add a second label with the same value, check if each step has the same sliced label
    test_labels[:, 5:8, 10:13] = 1
    sliced_labels = slice_labels(test_labels)
    assert np.all(np.unique(sliced_labels) == np.arange(6))
    for i in range(5):
        assert np.all(np.unique(sliced_labels[i]) == np.array([0, i + 1]))

    # Add a second label with a difference value, check if each step has two labels
    test_labels[:, 5:8, 10:13] = 2
    sliced_labels = slice_labels(test_labels)
    assert np.all(np.unique(sliced_labels) == np.arange(11))
    for i in range(5):
        assert np.all(
            np.unique(sliced_labels[i]) == np.array([0, (2 * i) + 1, (2 * i) + 2])
        )

    # Test that if a higher value label ends before a lower one, they all have unique values and are all one one step

    test_labels[1:3, 7:9, 2:5] = 3
    test_result = slice_labels(test_labels)
    label_wh_i = [
        np.unique(test_result[test_labels == 1]),
        np.unique(test_result[test_labels == 2]),
        np.unique(test_result[test_labels == 3]),
    ]
    assert len(np.intersect1d(label_wh_i[0], label_wh_i[1])) == 0
    assert len(np.intersect1d(label_wh_i[0], label_wh_i[2])) == 0
    assert len(np.intersect1d(label_wh_i[1], label_wh_i[2])) == 0

    for i in np.unique(test_result):
        if i > 0:
            wh_label_i_dim = np.where(test_result == i)[0]
            assert np.all(wh_label_i_dim == wh_label_i_dim[0])

def test_make_step_labels():
    from tobac_flow.utils.label_utils import make_step_labels

    test_labels = np.array([
        [
            [0, 0, 0, 1],
            [0, 2, 1, 0],
            [0, 2, 0, 3],
        ],
        [
            [0, 0, 0, 0],
            [0, 2, 2, 0],
            [0, 2, 0, 4],
        ],
    ])

    assert np.all(
        make_step_labels(test_labels) == np.array([
            [
                [0, 0, 0, 1],
                [0, 3, 2, 0],
                [0, 3, 0, 4]
            ],
            [
                [0, 0, 0, 0],
                [0, 5, 5, 0],
                [0, 5, 0, 6]
            ]
        ])
    )
