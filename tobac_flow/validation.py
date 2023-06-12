from datetime import datetime
import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from tobac_flow.utils import apply_func_to_labels

from tobac_flow.dataset import (
    add_dataarray_to_ds,
    create_dataarray,
)


def get_min_dist_for_objects(distance_array, labels, index=None):
    if isinstance(labels, xr.DataArray):
        labels = labels.to_numpy()
    if isinstance(distance_array, xr.DataArray):
        distance_array = distance_array.to_numpy()

    return apply_func_to_labels(
        labels, distance_array, func=np.nanmin, index=index, default=np.nan
    )


def get_marker_distance(labels, time_range=1):
    marker_distance = np.zeros(labels.shape)
    for i in range(marker_distance.shape[0]):
        if np.any(labels[i] != 0):
            marker_distance[i] = ndi.morphology.distance_transform_edt(labels[i] == 0)
        else:
            marker_distance[i] = np.inf

    for i in range(1, time_range + 1):
        marker_distance[i:] = np.fmin(marker_distance[:-i], marker_distance[i:])
        marker_distance[:-i] = np.fmin(marker_distance[:-i], marker_distance[i:])

    return marker_distance


def get_marker_distance_ellipse(markers, time_margin, margin):
    """
    Get distances to markers and nearest marker for each point in an ellipse
    with semi-major axis of length time_margin in the leading dimension and
    margin in the other dimensions
    """
    distances, indices = ndi.distance_transform_edt(
        markers == 0, return_indices=True, sampling=(margin / time_margin, 1, 1)
    )
    closest_marker = markers[indices[0], indices[1], indices[2]]
    return distances, closest_marker


def get_marker_distance_cylinder(markers, time_margin, get_closest=False):
    """
    Get distances to markers and nearest marker for each point in a cylinder
    with height time_margin in the leading dimension and radius of margin in
    the other dimensions
    """
    distances = np.full(markers.shape, np.inf, dtype=float)
    if get_closest:
        closest_markers = np.full(markers.shape, 0, dtype=int)

        for i in range(markers.shape[0]):
            if np.any(markers[i]):
                step_distances, indices = ndi.distance_transform_edt(
                    markers[i] == 0, return_indices=True
                )
                distances[i] = step_distances
                closest_markers[i] = markers[i][indices[0], indices[1]]

        distances2 = np.full(markers.shape, np.inf, dtype=float)
        closest_markers2 = np.full(markers.shape, 0, dtype=int)

        for i in range(markers.shape[0]):
            i_slice = slice(
                np.maximum(i - time_margin, 0),
                np.minimum(i + time_margin + 1, markers.shape[0]),
            )
            argmin = np.expand_dims(np.nanargmin(distances[i_slice], axis=0), axis=0)
            distances2[i] = np.take_along_axis(distances[i_slice], argmin, axis=0)
            closest_markers2[i] = np.take_along_axis(
                closest_markers[i_slice], argmin, axis=0
            )

        res = (distances2, closest_markers2)

    else:
        for i in range(markers.shape[0]):
            if np.any(markers[i]):
                step_distances = ndi.distance_transform_edt(markers[i] == 0)
                distances[i] = step_distances

        distances2 = np.full(markers.shape, np.inf, dtype=float)

        for i in range(markers.shape[0]):
            i_slice = slice(
                np.maximum(i - time_margin, 0),
                np.minimum(i + time_margin + 1, markers.shape[0]),
            )
            argmin = np.expand_dims(np.nanargmin(distances[i_slice], axis=0), axis=0)
            distances2[i] = np.take_along_axis(distances[i_slice], argmin, axis=0)

        res = distances2

    return res


def validate_markers(
    labels,
    glm_grid,
    glm_distance,
    edge_filter,
    n_glm_in_margin,
    coord=None,
    margin=10,
    time_margin=3,
    get_closest=False,
):
    """
    Get validation results for a set of markers
    """
    if get_closest:
        marker_distance, closest_marker = get_marker_distance_cylinder(
            labels, time_margin, get_closest=get_closest
        )
        flash_closest_marker = np.repeat(
            closest_marker.ravel(),
            glm_grid.astype(int).ravel(),
        )
    else:
        marker_distance = get_marker_distance_cylinder(
            labels, time_margin, get_closest=get_closest
        )
        flash_closest_marker = None
    flash_distance_to_marker = np.repeat(
        marker_distance.ravel(),
        glm_grid.astype(int).ravel(),
    )

    if n_glm_in_margin > 0:
        pod = np.nansum(flash_distance_to_marker <= margin) / n_glm_in_margin
    else:
        pod = np.nan

    margin_flag = apply_func_to_labels(
        labels, edge_filter, func=np.nanmin, index=coord, default=False
    ).astype("bool")

    n_marker_in_margin = np.nansum(margin_flag)

    marker_distance_to_flash = get_min_dist_for_objects(
        glm_distance, labels, index=coord
    )

    if n_marker_in_margin > 0:
        far = (
            np.nansum(marker_distance_to_flash[margin_flag] > margin)
            / n_marker_in_margin
        )
    else:
        far = np.nan

    return (
        flash_distance_to_marker,
        flash_closest_marker,
        marker_distance_to_flash,
        pod,
        far,
        n_marker_in_margin,
        margin_flag,
    )


def get_edge_filter(gridded_flash_ds, margin, time_margin):
    # Create an array to filter objects near to boundaries
    edge_filter_array = np.full(gridded_flash_ds.glm_flashes.shape, 1).astype("bool")

    # Filter edges
    edge_filter_array[:time_margin] = False
    edge_filter_array[-time_margin:] = False
    edge_filter_array[:, :margin] = False
    edge_filter_array[:, -margin:] = False
    edge_filter_array[:, :, :margin] = False
    edge_filter_array[:, :, -margin:] = False

    time_gap = np.where((np.diff(gridded_flash_ds.t) / 1e9).astype(int) > 900)[0]
    if time_gap.size > 0:
        print("Time gaps detected, filtering")
        for i in time_gap:
            i_slice = slice(
                np.maximum(i - time_margin + 1, 0),
                np.minimum(i + time_margin + 2, gridded_flash_ds.t.size),
            )
            edge_filter_array[i_slice] = False

    if np.any(gridded_flash_ds.glm_flashes == -1):
        print("Missing glm data detected, filtering")
        margin_structure = np.stack(
            [
                np.sum(
                    [
                        (arr - 10) ** 2
                        for arr in np.meshgrid(
                            np.arange(margin * 2 + 1), np.arange(margin * 2 + 1)
                        )
                    ],
                    0,
                )
                ** 0.5
                < margin
            ]
            * (time_margin * 2 + 1),
            0,
        )
        wh_missing_glm = ndi.binary_dilation(
            gridded_flash_ds.glm_flashes.to_numpy() == -1, structure=margin_structure
        )
        edge_filter_array[wh_missing_glm] = False

    return edge_filter_array


def validate_cores(
    detection_ds,
    validation_ds,
    glm_grid,
    glm_distance,
    edge_filter_array,
    n_glm_in_margin,
    margin,
    time_margin,
    get_closest=False,
):
    core_label = detection_ds.core_label.to_numpy()
    core_coord = detection_ds.core.to_numpy()

    (
        flash_distance_to_core,
        flash_nearest_core,
        core_min_distance,
        core_pod,
        core_far,
        n_core_in_margin,
        core_margin_flag,
    ) = validate_markers(
        core_label,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=core_coord,
        margin=margin,
        time_margin=time_margin,
        get_closest=get_closest,
    )

    print("cores:", flush=True)
    print("n =", n_core_in_margin, flush=True)
    print("POD =", core_pod, flush=True)
    print("FAR = ", core_far, flush=True)

    # Write to dataset
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_core,
            ("flash",),
            "flash_core_distance",
            long_name="closest distance from flash to detected core",
            dtype=np.float32,
        ),
        validation_ds,
    )
    if flash_nearest_core is not None:
        add_dataarray_to_ds(
            create_dataarray(
                flash_nearest_core,
                ("flash",),
                "flash_core_index",
                long_name="index of nearest detected core to each flash",
                dtype=np.int32,
            ),
            validation_ds,
        )
    add_dataarray_to_ds(
        create_dataarray(
            core_min_distance,
            ("core",),
            "core_glm_distance",
            long_name="closest distance from core to GLM flash",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_pod, tuple(), "core_pod", long_name="POD for cores", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_far, tuple(), "core_far", long_name="FAR for cores", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_core_in_margin,
            tuple(),
            "core_count_in_margin",
            long_name="total number of cores inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_margin_flag,
            ("core",),
            "core_margin_flag",
            long_name="margin flag for core",
            dtype=bool,
        ),
        validation_ds,
    )


def validate_cores_with_anvils(
    detection_ds,
    validation_ds,
    glm_grid,
    glm_distance,
    edge_filter_array,
    n_glm_in_margin,
    margin,
    time_margin,
    get_closest=False,
):
    core_with_anvil_coord = detection_ds.core.to_numpy()[
        detection_ds.core_anvil_index.to_numpy() != 0
    ]
    core_with_anvil_label = detection_ds.core_label.to_numpy() * np.isin(
        detection_ds.core_label.to_numpy(), core_with_anvil_coord
    ).astype(int)

    (
        flash_distance_to_core_with_anvil,
        flash_nearest_core_with_anvil,
        core_with_anvil_min_distance,
        core_with_anvil_pod,
        core_with_anvil_far,
        n_core_with_anvil_in_margin,
        core_with_anvil_margin_flag,
    ) = validate_markers(
        core_with_anvil_label,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=core_with_anvil_coord,
        margin=margin,
        time_margin=time_margin,
        get_closest=get_closest,
    )

    print("cores with anvils:", flush=True)
    print("n =", n_core_with_anvil_in_margin, flush=True)
    print("POD =", core_with_anvil_pod, flush=True)
    print("FAR = ", core_with_anvil_far, flush=True)

    # Write to dataset
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_core_with_anvil,
            ("flash",),
            "flash_core_with_anvil_distance",
            long_name="closest distance from flash to detected core_with_anvil",
            dtype=np.float32,
        ),
        validation_ds,
    )
    if flash_distance_to_core_with_anvil is not None:
        add_dataarray_to_ds(
            create_dataarray(
                flash_nearest_core_with_anvil,
                ("flash",),
                "flash_core_with_anvil_index",
                long_name="index of nearest detected core_with_anvil to each flash",
                dtype=np.int32,
            ),
            validation_ds,
        )
    add_dataarray_to_ds(
        create_dataarray(
            core_with_anvil_min_distance,
            ("core_with_anvil",),
            "core_with_anvil_glm_distance",
            long_name="closest distance from core_with_anvil to GLM flash",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_with_anvil_pod,
            tuple(),
            "core_with_anvil_pod",
            long_name="POD for core_with_anvils",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_with_anvil_far,
            tuple(),
            "core_with_anvil_far",
            long_name="FAR for core_with_anvils",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_core_with_anvil_in_margin,
            tuple(),
            "core_with_anvil_count_in_margin",
            long_name="total number of core_with_anvils inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_with_anvil_margin_flag,
            ("core_with_anvil",),
            "core_with_anvil_margin_flag",
            long_name="margin flag for core_with_anvil",
            dtype=bool,
        ),
        validation_ds,
    )


def validate_anvils(
    detection_ds,
    validation_ds,
    glm_grid,
    glm_distance,
    edge_filter_array,
    n_glm_in_margin,
    margin,
    time_margin,
    get_closest=False,
):
    thick_anvil_label = detection_ds.thick_anvil_label.to_numpy()
    anvil_coord = detection_ds.anvil.to_numpy()

    (
        flash_distance_to_anvil,
        flash_nearest_anvil,
        anvil_min_distance,
        anvil_pod,
        anvil_far,
        n_anvil_in_margin,
        anvil_margin_flag,
    ) = validate_markers(
        thick_anvil_label,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=anvil_coord,
        margin=margin,
        time_margin=time_margin,
        get_closest=get_closest,
    )

    print("anvil:", flush=True)
    print("n =", n_anvil_in_margin, flush=True)
    print("POD =", anvil_pod, flush=True)
    print("FAR = ", anvil_far, flush=True)

    # Write to dataset
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_anvil,
            ("flash",),
            "flash_anvil_distance",
            long_name="closest distance from flash to detected anvil",
            dtype=np.float32,
        ),
        validation_ds,
    )
    if flash_nearest_anvil is not None:
        add_dataarray_to_ds(
            create_dataarray(
                flash_nearest_anvil,
                ("flash",),
                "flash_anvil_index",
                long_name="index of nearest detected anvil to each flash",
                dtype=np.int32,
            ),
            validation_ds,
        )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_min_distance,
            ("anvil",),
            "anvil_glm_distance",
            long_name="closest distance from anvil to GLM flash",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_pod,
            tuple(),
            "anvil_pod",
            long_name="POD for anvils",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_far,
            tuple(),
            "anvil_far",
            long_name="FAR for anvils",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_anvil_in_margin,
            tuple(),
            "anvil_count_in_margin",
            long_name="total number of anvils inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_margin_flag,
            ("anvil",),
            "anvil_margin_flag",
            long_name="margin flag for anvil",
            dtype=bool,
        ),
        validation_ds,
    )


def validate_anvils_with_cores(
    detection_ds,
    validation_ds,
    glm_grid,
    glm_distance,
    edge_filter_array,
    n_glm_in_margin,
    margin,
    time_margin,
    get_closest=False,
):
    anvil_with_core_coord = detection_ds.anvil.to_numpy()[
        np.isin(detection_ds.anvil.to_numpy(), detection_ds.core_anvil_index.to_numpy())
    ]
    anvil_with_core_label = detection_ds.thick_anvil_label.to_numpy() * np.isin(
        detection_ds.thick_anvil_label.to_numpy(), anvil_with_core_coord
    ).astype(int)

    (
        flash_distance_to_anvil_with_core,
        flash_nearest_anvil_with_core,
        anvil_with_core_min_distance,
        anvil_with_core_pod,
        anvil_with_core_far,
        n_anvil_with_core_in_margin,
        anvil_with_core_margin_flag,
    ) = validate_markers(
        anvil_with_core_label,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=anvil_with_core_coord,
        margin=margin,
        time_margin=time_margin,
        get_closest=get_closest,
    )

    print("anvil with cores:", flush=True)
    print("n =", n_anvil_with_core_in_margin, flush=True)
    print("POD =", anvil_with_core_pod, flush=True)
    print("FAR = ", anvil_with_core_far, flush=True)

    # Write to dataset
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_anvil_with_core,
            ("flash",),
            "flash_anvil_with_core_distance",
            long_name="closest distance from flash to detected anvil_with_core",
            dtype=np.float32,
        ),
        validation_ds,
    )
    if flash_nearest_anvil_with_core is not None:
        add_dataarray_to_ds(
            create_dataarray(
                flash_nearest_anvil_with_core,
                ("flash",),
                "flash_anvil_with_core_index",
                long_name="index of nearest detected anvil_with_core to each flash",
                dtype=np.int32,
            ),
            validation_ds,
        )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_with_core_min_distance,
            ("anvil_with_core",),
            "anvil_with_core_glm_distance",
            long_name="closest distance from anvil_with_core to GLM flash",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_with_core_pod,
            tuple(),
            "anvil_with_core_pod",
            long_name="POD for anvil_with_cores",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_with_core_far,
            tuple(),
            "anvil_with_core_far",
            long_name="FAR for anvil_with_cores",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_anvil_with_core_in_margin,
            tuple(),
            "anvil_with_core_count_in_margin",
            long_name="total number of anvil_with_cores inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_with_core_margin_flag,
            ("anvil_with_core",),
            "anvil_with_core_margin_flag",
            long_name="margin flag for anvil_with_core",
            dtype=bool,
        ),
        validation_ds,
    )


def validate_anvil_markers(
    detection_ds,
    validation_ds,
    glm_grid,
    glm_distance,
    edge_filter_array,
    n_glm_in_margin,
    margin,
    time_margin,
    get_closest=False,
):
    anvil_marker_label = detection_ds.anvil_marker_label.to_numpy()
    anvil_marker_coord = detection_ds.anvil_marker.to_numpy()

    (
        flash_distance_to_anvil_marker,
        flash_nearest_anvil_marker,
        anvil_marker_min_distance,
        anvil_marker_pod,
        anvil_marker_far,
        n_anvil_marker_in_margin,
        anvil_marker_margin_flag,
    ) = validate_markers(
        anvil_marker_label,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=anvil_marker_coord,
        margin=margin,
        time_margin=time_margin,
        get_closest=get_closest,
    )

    print("anvil marker:", flush=True)
    print("n =", n_anvil_marker_in_margin, flush=True)
    print("POD =", anvil_marker_pod, flush=True)
    print("FAR = ", anvil_marker_far, flush=True)

    # Write to dataset
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_anvil_marker,
            ("flash",),
            "flash_anvil_marker_distance",
            long_name="closest distance from flash to detected anvil_marker",
            dtype=np.float32,
        ),
        validation_ds,
    )
    if flash_nearest_anvil_marker is not None:
        add_dataarray_to_ds(
            create_dataarray(
                flash_nearest_anvil_marker,
                ("flash",),
                "flash_anvil_marker_index",
                long_name="index of nearest detected anvil_marker to each flash",
                dtype=np.int32,
            ),
            validation_ds,
        )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_marker_min_distance,
            ("anvil_marker",),
            "anvil_marker_glm_distance",
            long_name="closest distance from anvil_marker to GLM flash",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_marker_pod,
            tuple(),
            "anvil_marker_pod",
            long_name="POD for anvil_markers",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_marker_far,
            tuple(),
            "anvil_marker_far",
            long_name="FAR for anvil_markers",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_anvil_marker_in_margin,
            tuple(),
            "anvil_marker_count_in_margin",
            long_name="total number of anvil_markers inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_marker_margin_flag,
            ("anvil_marker",),
            "anvil_marker_margin_flag",
            long_name="margin flag for anvil_marker",
            dtype=bool,
        ),
        validation_ds,
    )
