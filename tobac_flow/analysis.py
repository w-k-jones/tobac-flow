import numpy as np
from scipy import ndimage as ndi
from .dataset import add_dataarray_to_ds

def apply_func_to_labels(labels, field, func, dtype=None):
    if dtype == None:
        dtype = field.dtype
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    return np.array([func(field.ravel()[args[bins[i]:bins[i+1]]].astype(dtype))
                     if bins[i+1]>bins[i] else None for i in range(bins.size-1)])

def apply_weighted_func_to_labels(labels, field, weights, func, dtype=None):
    if dtype == None:
        dtype = field.dtype
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    return np.array([func(field.ravel()[args[bins[i]:bins[i+1]]].astype(dtype),
                          weights.ravel()[args[bins[i]:bins[i+1]]])
                     if bins[i+1]>bins[i] else None for i in range(bins.size-1)])

def flat_label(mask, structure=ndi.generate_binary_structure(3,1),
               dtype=np.int32):
    label_struct = structure.copy()
    label_struct[0] = 0
    label_struct[-1] = 0

    return ndi.label(mask, structure=label_struct, output=dtype)[0]

def get_step_labels_for_label(labels, structure=ndi.generate_binary_structure(3,1),
                              dtype=np.int32):
    step_labels = flat_label(labels!=0, structure=structure, dtype=dtype)
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    return [np.unique(step_labels.ravel()[args[bins[i]:bins[i+1]]])
            if bins[i+1]>bins[i] else None for i in range(bins.size-1)]

def relabel_objects(labels):
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    new_labels = np.zeros_like(labels)
    counter = 1
    for i in range(bins.size-1):
        if bins[i+1]>bins[i]:
            new_labels.ravel()[args[bins[i]:bins[i+1]]] = counter
            counter += 1
    return new_labels

def slice_labels(labels):
    step_labels = np.zeros_like(labels)
    max_label = 0
    for i in range(labels.shape[0]):
        step_labels[i] = relabel_objects(labels[i])
        step_labels[i][np.nonzero(step_labels[i])] += max_label
        max_label = step_labels.max()
    return step_labels

def slice_label_da(label_da):
    label_name = label_da.name.split("_label")[0]
    step_labels = create_dataarray(slice_labels(label_da.data), label_da.dims, f"{label_name}_step_label",
                                   long_name=f"{label_da.long_name} at each time step", units="", dtype=np.int32)
    return step_labels

def filter_labels_by_length(labels, min_length):
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    object_lengths = np.array([o[0].stop-o[0].start for o in ndi.find_objects(labels)])
    counter = 1
    for i in range(bins.size-1):
        if bins[i+1]>bins[i]:
            if object_lengths[i]<min_length:
                labels.ravel()[args[bins[i]:bins[i+1]]] = 0
            else:
                labels.ravel()[args[bins[i]:bins[i+1]]] = counter
                counter += 1
    return labels

def filter_labels_by_length_and_mask(labels, mask, min_length):
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    object_lengths = np.array([o[0].stop-o[0].start for o in ndi.find_objects(labels)])
    counter = 1
    for i in range(bins.size-1):
        if bins[i+1]>bins[i]:
            if object_lengths[i]>=min_length and np.any(mask.ravel()[args[bins[i]:bins[i+1]]]):
                labels.ravel()[args[bins[i]:bins[i+1]]] = counter
                counter += 1
            else:
                labels.ravel()[args[bins[i]:bins[i+1]]] = 0
    return labels

def filter_labels_by_length_and_multimask(labels, masks, min_length):
    if type(masks) is not type(list()):
        raise ValueError("masks input must be a list of masks to process")

    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    object_lengths = np.array([o[0].stop-o[0].start for o in ndi.find_objects(labels)])
    counter = 1
    for i in range(bins.size-1):
        if bins[i+1]>bins[i]:
            if object_lengths[i]>=min_length and np.all([np.any(m.ravel()[args[bins[i]:bins[i+1]]]) for m in masks]):
                labels.ravel()[args[bins[i]:bins[i+1]]] = counter
                counter += 1
            else:
                labels.ravel()[args[bins[i]:bins[i+1]]] = 0
    return labels

def get_stats_for_labels(labels, da, dim=None, dtype=None):
    if not dim:
        dim = labels.name.split("_label")[0]
    if dtype == None:
        dtype = da.dtype
    mean_da = create_dataarray(apply_func_to_labels(labels.data, da.data, np.nanmean, dtype=dtype),
                               (dim,), f"{dim}_{da.name}_mean",
                               long_name=f"Mean of {da.long_name} for each {dim}",
                               units=da.units, dtype=dtype)
    std_da = create_dataarray(apply_func_to_labels(labels.data, da.data, np.nanstd, dtype=dtype),
                              (dim,), f"{dim}_{da.name}_std",
                              long_name=f"Standard deviation of {da.long_name} for each {dim}",
                              units=da.units, dtype=dtype)
    max_da = create_dataarray(apply_func_to_labels(labels.data, da.data, np.nanmax, dtype=dtype),
                              (dim,), f"{dim}_{da.name}_max",
                              long_name=f"Maximum of {da.long_name} for each {dim}",
                              units=da.units, dtype=dtype)
    min_da = create_dataarray(apply_func_to_labels(labels.data, da.data, np.nanmin, dtype=dtype),
                              (dim,), f"{dim}_{da.name}_min",
                              long_name=f"Minimum of {da.long_name} for each {dim}",
                              units=da.units, dtype=dtype)

    return mean_da, std_da, max_da, min_da

from .dataset import create_dataarray, n_unique_along_axis
def get_label_stats(da, ds):
    add_dataarray_to_ds(create_dataarray(np.count_nonzero(da, 0)/da.t.size, ('y', 'x'),
                                         f"{da.name}_fraction",
                                         long_name=f"Fractional coverage of {da.long_name}",
                                         units="", dtype=np.float32), ds)
    add_dataarray_to_ds(create_dataarray(n_unique_along_axis(da.data, 0), ('y', 'x'),
                                         f"{da.name}_unique_count",
                                         long_name=f"Number of unique {da.long_name}",
                                         units="", dtype=np.int32), ds)

    add_dataarray_to_ds(create_dataarray(np.count_nonzero(da, (1,2))/(da.x.size*da.y.size), ('t',),
                                         f"{da.name}_temporal_fraction",
                                         long_name=f"Fractional coverage of {da.long_name} over time",
                                         units="", dtype=np.float32), ds)
    add_dataarray_to_ds(create_dataarray(n_unique_along_axis(da.data.reshape([da.t.size,-1]), 1), ('t',),
                                         f"{da.name}_temporal_unique_count",
                                         long_name=f"Number of unique {da.long_name} over time",
                                         units="", dtype=np.int32), ds)
