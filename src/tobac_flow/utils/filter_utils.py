"""
Utilities for filtering various detected features
"""
import numpy as np
import xarray as xr


def remove_orphan_coords(dataset: xr.Dataset) -> xr.Dataset:
    """
    Remove cores/anvils which don't have core/anvil steps and vice versa
    """
    wh_core = np.isin(dataset.core, dataset.core_step_core_index)
    wh_anvil = np.logical_and(
        np.isin(dataset.anvil, dataset.thick_anvil_step_anvil_index),
        np.isin(dataset.anvil, dataset.thin_anvil_step_anvil_index),
    )
    dataset = dataset.sel(
        core=dataset.core.data[wh_core], anvil=dataset.anvil.data[wh_anvil]
    )
    wh_core_step = np.isin(dataset.core_step_core_index, dataset.core)
    wh_thick_anvil_step = np.isin(dataset.thick_anvil_step_anvil_index, dataset.anvil)
    wh_thin_anvil_step = np.isin(dataset.thin_anvil_step_anvil_index, dataset.anvil)
    dataset = dataset.sel(
        core_step=dataset.core_step.data[wh_core_step],
        thick_anvil_step=dataset.thick_anvil_step[wh_thick_anvil_step],
        thin_anvil_step=dataset.thin_anvil_step[wh_thin_anvil_step],
    )
    return dataset
