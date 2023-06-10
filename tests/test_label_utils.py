import numpy as np
from tobac_flow.utils import label_utils


def test_apply_func_to_labels():
    test_labels = np.zeros([4, 6])
    test_labels[1:3, 1:3] = 1
    test_labels[2:3, 3:6] = 3
    test_labels = test_labels.astype(int)
    test_data1 = np.arange(24).reshape([4, 6])
    test_data2 = np.array([1, 2, 3, 3, 2, 1])

    label_utils.apply_func_to_labels(test_labels, test_data1, func=np.mean)
    label_utils.apply_func_to_labels(
        test_labels, np.stack([test_data1, test_data1]), func=np.mean
    )
    label_utils.apply_func_to_labels(
        test_labels,
        np.stack([test_data1, test_data1]),
        func=np.mean,
        index=[1, 3],
    )
    label_utils.apply_func_to_labels(
        test_labels,
        np.stack([test_data1, test_data1]),
        func=np.mean,
        index=[1, 2, 3, 4],
    )
    label_utils.apply_func_to_labels(
        test_labels,
        np.stack([test_data1, test_data1]),
        func=np.mean,
        index=[0, 1, 2, 3, 4],
    )
    label_utils.apply_func_to_labels(
        test_labels,
        np.stack([test_data1, test_data1]),
        func=np.mean,
        index=[-1, 1, 2, 3, 4],
    )


    weighted_mean = lambda a, w: np.average(a, weights=w)
    label_utils.apply_func_to_labels(test_labels, test_data1, 1, func=weighted_mean)
    label_utils.apply_func_to_labels(
        test_labels, test_data1, test_data2, func=weighted_mean
    )
    label_utils.apply_func_to_labels(
        test_labels, np.stack([test_data1, test_data1]), 1, func=weighted_mean
    )

    mean_and_std = lambda a: (np.mean(a), np.std(a))
    label_utils.apply_func_to_labels(test_labels, test_data1, func=mean_and_std)

    weighted_mean_and_std = lambda a, w: (np.average(a, weights=w), np.std(a))
    label_utils.apply_func_to_labels(
        test_labels, test_data1, test_data2, func=weighted_mean_and_std
    )

    label_utils.apply_func_to_labels(
        test_labels, test_data1, test_data2, func=weighted_mean_and_std, default=np.nan
    )

    label_utils.apply_func_to_labels(
        test_labels,
        test_data1,
        test_data2,
        func=weighted_mean_and_std,
        default=[np.nan] * 2,
    )

    label_utils.apply_func_to_labels(
        test_labels, test_data1, test_data2, func=weighted_mean, default="nan"
    )

    label_utils.apply_func_to_labels(
        test_labels, test_data1, test_data2, func=weighted_mean, default=["nan"]
    )
