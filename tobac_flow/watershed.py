"""
Watershed segmentation operations in a semi-Lagrangian framework.
This uses a legacy version of the tobac_flow framework for handling the
    convolutions. I would like to update this but it's currently faster to use
    this version rather than the more general versions used in the flow object
"""

import cv2 as cv
import numpy as np
from numpy import ma
import xarray as xr
from scipy import interpolate
from scipy import ndimage as ndi

class Flow_Func(object):
    """
    Legacy flow object. Holds forwards and backwards flow vectors for use in
        legacy semi-Lagrangian operations
    """
    def __init__(self, flow_x_for, flow_x_back, flow_y_for, flow_y_back):
        """
        Initiate legacy flow object

        Parameters
        ----------
        flow_x_for : numpy.ndarray
            Array of the forward flow vectors in the x direction

        flow_x_back : numpy.ndarray
            Array of the backward flow vectors in the x direction

        flow_y_for : numpy.ndarray
            Array of the forward flow vectors in the y direction

        flow_y_back : numpy.ndarray
            Array of the backward flow vectors in the y direction

        TODO : Generalise this for n dimensions (3d support priority)
        """
        self.flow_x_for = flow_x_for
        self.flow_y_for = flow_y_for
        self.flow_x_back = flow_x_back
        self.flow_y_back = flow_y_back
        self.shape = flow_x_for.shape

    def __getitem__(self, items):
        """
        return a subset of the flow vectors
        """
        return Flow_Func(self.flow_x_for[items], self.flow_x_back[items],
                         self.flow_y_for[items], self.flow_y_back[items])

    def __call__(self, t):
        """
        parabolic interpolation of the flow vectors

        Parameters
        ----------
        t : float
            time step to provide flow vectors at. Accurate 1 and -1

        Returns
        -------
        tuple (ndarray, ndarray) :
            A tuple of arrays of the optical flow offset vectors in the x and y
            directions.

        TODO : generalise for nd
        """
        return (0.5*t*(t+1)*self.flow_x_for + 0.5*t*(t-1)*self.flow_x_back,
                0.5*t*(t+1)*self.flow_y_for + 0.5*t*(t-1)*self.flow_y_back)

def _checkstruct(structure, n_dims):
    """
    Check if an input structuring element is valid. i.e. it does not have more
        dimensions than the data it is to be applied to, and each dimension is
        either of length 1 or 3

    Parameters
    ----------
    structure : numpy.ndarray
        The structuring array to test

    ndims : int
        The number of dimensions of the data that the structuring element is to
            be applied to

    Returns
    -------
    structure : numpy.ndarray
        Structuring array that has met the criteria and is rehshaped to have the
            same number of dimenions as the data it is being applied to
    """
    if structure is None:
        structure = ndi.generate_binary_structure(n_dims,1)

    if hasattr(structure, "shape"):

        if len(structure.shape) > n_dims:
            raise ValueError("Input structure has too many dimensions")

        for s in structure.shape:
            if s not in [1,3]:
                raise ValueError("""structure input must be an array with
                                    dimensions of length 1 or 3""")
        if len(structure.shape) < n_dims:
            nd_diff = n_dims - len(structure.shape)
            structure = structure.reshape((1,)*nd_diff+structure.shape)

    else:
        raise ValueError("structure input must be an array-like object")

    return structure

def _gen_flow_ravel_inds(flow_func, structure, wrap=False):
    """
    Creates a generator for the indexes of the nearest neighbour pixels for
        convolutions at each time step

    Parameters
    ----------
    flow_func : object
        Flow_Func object from this module

    structure : numpy.ndarray
        Structuring element for the convolution

    Returns
    -------
    ravelled_index : numpy.ndarray
        Array indexes of the convolved pixels for the chosen structure at each
            time step.

    mask : numpy.ndarray
        Array of the values at the edge of the array to be masked in the
            convolution. If the 'wrap' keyword is true then this will return
            False as all values are included.

    Keywords
    --------
    wrap : bool (default: False)
        If true, wrap values at the edge of the array according to periodic
            boundary conditions. Defaults to false
    """
    shape = flow_func.shape
    n_dims = len(shape)

    structure_offsets = [wh.reshape((-1,)+(1,)*(n_dims-1)) - 1
                         for wh in np.where(structure!=0)]
    wh_forward = (structure_offsets[0] == 1)
    wh_backward = (structure_offsets[0] == -1)
    n_elements = np.sum(structure!=0)

    shape_ranges = [np.arange(s).reshape(np.roll((-1,)+(1,)*(n_dims-2), i))
                    for i, s in enumerate(shape[1:])]

    flow_inds = [shape_ranges[i] + structure_offsets[i+1] for i in range(n_dims-1)]
    for t in range(shape[0]):
        # Todo: make this generalised for more dimensions
        temp_inds = [None, None]
        temp_inds[0] = (flow_inds[0]
                        + np.round(flow_func.flow_y_for[t]).astype(int) * wh_forward
                        + np.round(flow_func.flow_y_back[t]).astype(int) * wh_backward)
        temp_inds[1] = (flow_inds[1]
                        + np.round(flow_func.flow_x_for[t]).astype(int) * wh_forward
                        + np.round(flow_func.flow_x_back[t]).astype(int) * wh_backward)

        ravelled_index = np.ravel_multi_index([(structure_offsets[0]+t) % shape[0],
                                               temp_inds[0] % shape[1],
                                               temp_inds[1] % shape[2]],
                                              shape).ravel()
        if wrap:
            mask = False
        else:
            mask = sum([(structure_offsets[0]+t)%shape[0] != (structure_offsets[0]+t),
                        (temp_inds[0]%shape[1]) != temp_inds[0],
                        (temp_inds[1]%shape[2]) != temp_inds[1]])

        yield ravelled_index, mask

def find_nearest_neighbour_inds(data, flow_func, structure=None, wrap=False, dtype=None):

    if dtype == None:
        dtype = data.dtype
    n_dims = len(data.shape)
    assert(n_dims > 1)

    structure = _checkstruct(structure, n_dims)
    structure_factor = structure[structure!=0].reshape((-1,)+(1,)*(n_dims-1))
    n_elements = np.sum(structure!=0)

    inds_gen = _gen_flow_ravel_inds(flow_func, structure, wrap=wrap)

    out_arr = np.empty(data.shape, dtype=dtype)

    for t in range(data.shape[0]):
        ravelled_index, mask = next(inds_gen)

        temp = ma.array(data.ravel()[ravelled_index].reshape((n_elements,)+data.shape[1:]) * structure_factor,
                        mask=mask, dtype=data.dtype)

        argmin = ma.argmin(temp, 0)

        out_arr[t] = np.take_along_axis(ravelled_index.reshape((n_elements,)+data.shape[1:]),
                                        argmin[np.newaxis,...],
                                        axis=0)

    return out_arr

def flow_network_watershed(field, markers, flow_func, mask=None, structure=None, max_iter=100, debug_mode=False, low_memory=False):
    # Check structure input, set default and check dimensions and shape
    n_dims = len(field.shape)
    assert(n_dims > 1)

    structure = _checkstruct(structure, n_dims)

    if np.any([s != 3 for s in structure.shape]):
        if debug_mode:
            print("Inserting structure into 3x3x3 array")
        wh = [slice(0,3) if s==3 else slice(1,2) for s in structure.shape]
        temp = np.zeros([3,3,3])
        temp[wh] = structure
        structure=temp

    if isinstance(structure, ma.core.MaskedArray):
        structure = structure.filled(fill_value=0)
    structure = structure.astype('bool')

    # Check mask input
    if mask is None:
        if debug_mode:
            print("Setting mask to default")
        mask = np.zeros_like(field, dtype='bool')
    if isinstance(mask, ma.core.MaskedArray):
        mask = mask.filled(fill_value=True)

    # Check markers input
    if isinstance(markers, ma.core.MaskedArray):
        markers = markers.filled(fill_value=False)

    # Check field input
    if isinstance(field, ma.core.MaskedArray):
        field = field.filled(fill_value=np.nanmax(field))

    # Check for NaN values in the input and set mask/marker values appropriately
    wh = np.isnan(field)
    if np.any(wh):
        field[wh] = np.nanmax(field)
        mask[wh] = True
        markers[wh] = False

    # Get ravelled indices for each pixel in the field, and find nearest neighbours using flow field
    # Set inds dtype to minimum possible to contain all values to save memory
    if field.size<np.iinfo(np.uint16).max:
        inds_dtype = np.uint16

    elif field.size<np.iinfo(np.uint32).max:
        inds_dtype = np.uint32

    else:
        inds_dtype = np.uint64

    inds = np.arange(field.size, dtype=inds_dtype).reshape(field.shape)

    if debug_mode:
        print("Calculating nearest neighbours")

    inds_neighbour = find_nearest_neighbour_inds(field,
                                                 flow_func,
                                                 structure=structure,
                                                 dtype=inds_dtype)

    # inds_neighbour = inds_convolve[tuple([min_convolve.data.astype(int)]+np.meshgrid(*(range(s) for s in inds.shape), indexing='ij'))].astype(int)
    if hasattr(inds_neighbour, "mask"):
        wh = np.logical_or.reduce([inds_neighbour.data<0,
                                   inds_neighbour.data>inds.max(),
                                   inds_neighbour.mask])
        if np.any(wh):
            inds_neighbour.data[wh] = inds[wh]
    else:
        wh = np.logical_or(inds_neighbour<0, inds_neighbour>inds.max())
        if np.any(wh):
            inds_neighbour[wh] = inds[wh]

    inds_neighbour = inds_neighbour.astype(inds_dtype)

    # Now iterate over neighbour network to find minimum convergence point for each pixel
        # Each pixel will either reach a minimum or loop back to itself
    if markers.max()<np.iinfo(np.int16).max:
        mark_dtype = np.int16
    elif markers.max()<np.iinfo(np.int32).max:
        mark_dtype = np.int32
    else:
        mark_dtype = np.int64

    fill_markers = markers.astype(mark_dtype) - mask.astype(mark_dtype)
    wh_local_min = np.logical_and(inds_neighbour==inds, fill_markers==0)
    wh_markers = np.logical_or(wh_local_min, fill_markers!=0)
    wh_to_fill = np.logical_not(wh_markers.copy()) # Do we need to copy this?
    if debug_mode:
        print("Finding network convergence locations")
        print("Pixels to fill:", np.sum(wh_to_fill))
    for i in range(max_iter):
        inds_neighbour[wh_to_fill] = inds_neighbour.ravel()[inds_neighbour[wh_to_fill].ravel()]
        # Check if any pixels have looped back to their original location
        wh_loop = np.logical_and(wh_to_fill, inds_neighbour==inds)
        if np.any(wh_loop):
            if debug_mode:
                print('Loop')
            wh_to_fill[wh_loop] = False
            wh_local_min[wh_loop] = True
            wh_markers[wh_loop] = True

        # Now check if any have met a convergence location
        wh_converge = wh_markers.ravel()[inds_neighbour[wh_to_fill]].ravel()
        if np.any(wh_converge):
            if debug_mode:
                print('Convergence')
            wh_to_fill[wh_to_fill] = np.logical_not(wh_converge)

        if debug_mode:
            print("Iteration:", i+1)
            print("Pixels converged", np.sum(np.logical_not(wh_to_fill)))
        if not np.any(wh_to_fill):
            if debug_mode:
                print("All pixels converged")
            break
    # del old_neighbour
    # Use converged locations to fill watershed basins
    if debug_mode:
        print("Filling basins")
    # wh = np.logical_and(type_converge==1, np.logical_not(np.logical_xor(markers!=0, mask)))
    max_markers = np.nanmax(markers)
    temp_markers = ndi.label(wh_local_min)[0][wh_local_min]+max_markers
    if np.any(wh_local_min):
        max_temp_marker = temp_markers.max()
    else:
        max_temp_marker = max_markers
    if max_temp_marker<np.iinfo(np.int16).max:
        mark_dtype = np.int16
    elif max_temp_marker<np.iinfo(np.int32).max:
        mark_dtype = np.int32
    else:
        mark_dtype = np.int64
    fill_markers = fill_markers.astype(mark_dtype)
    fill_markers[wh_local_min] = temp_markers
    fill = fill_markers.copy()
    wh = fill==0
    fill[wh] = fill.ravel()[inds_neighbour[wh].ravel()]
    # fill = fill_markers.ravel()[inds_neighbour.ravel()].reshape(fill_markers.shape)
    del fill_markers, temp_markers, inds_neighbour
    # fill[markers>0]=markers[markers>0]
    # fill[mask]=-1
    wh = fill==0
    if np.any(wh):
        if debug_mode:
            print("Some pixels not filled, adding")
        fill[wh] = ndi.label(wh)[0][wh]+np.nanmax(fill)
    # Now we've filled all the values, we change the mask values back to 0 for the next step
    if isinstance(fill, ma.core.MaskedArray):
        fill = np.maximum(fill.filled(fill_value=0),0)
    else:
        fill = np.maximum(fill, 0)
    # Now overflow watershed basins into neighbouring basins until only marker labels are left
    if debug_mode:
        print("Joining labels")
        print("Max label:", np.nanmax(fill))
        print("max_markers:", max_markers.astype(int))
    # we can set the middle value of the structure to 0 as we are only interested in the surrounding pixels
    new_struct = structure.copy()
    new_struct[1,1,1] = 0

    overflow_map = {}

    structure_offsets = np.stack([arr.reshape((-1,)+(1,)*(n_dims-2))-1 for arr in np.where(new_struct!=0)], 1)
    whp1 = structure_offsets[0] == 1
    whm1 = structure_offsets[0] == -1
    n_elements = np.sum(new_struct!=0)

    bins = np.cumsum(np.bincount(fill.ravel()))
    args = np.argsort(fill.ravel())

    shape = fill.shape
    strides = (shape[1]*shape[2], shape[2], 1)

    def get_field_minmax(field_max, field_min, wh):
        args = np.argsort(field_min[wh])
        wh_minmax = args[np.argmin(field_max[wh][args])]

        return [field_max[wh_minmax], field_min[wh_minmax]]

    for i in range(max_markers+1, fill.max()):

        inds = args[bins[i]:bins[i+1]]
        inds_stack = np.stack(np.unravel_index(inds, shape),0)
        temp_inds = inds_stack + structure_offsets
        wrapped_inds = temp_inds % np.array(shape)[np.newaxis,:,np.newaxis]
        mask = np.any(wrapped_inds != temp_inds, 1)

        ravelled_inds = np.ravel_multi_index([wrapped_inds[:,0], wrapped_inds[:,1], wrapped_inds[:,2]], shape)

        temp_fill = fill.ravel()[ravelled_inds][~mask]
#         temp_field = np.maximum(field.ravel()[ravelled_inds], field.ravel()[inds])[~mask]

#         overflow_map[i] = {j:np.nanmin(temp_field[temp_fill==j]) for j in np.unique(temp_fill) if j < i}

        temp_field_max = np.maximum(field.ravel()[ravelled_inds], field.ravel()[inds])[~mask]
        temp_field_min = np.minimum(field.ravel()[ravelled_inds], field.ravel()[inds])[~mask]

        overflow_map[i] = {j:get_field_minmax(temp_field_max, temp_field_min, temp_fill==j)
                           for j in np.unique(temp_fill) if j < i}

    key_map = np.zeros(fill.max()+1, dtype=int)
    key_map[:max_markers+1] = np.arange(max_markers+1)

    def get_key_minmax(keys, overflow_list):
        overflow_list = np.array(list(overflow_list))
        args = np.argsort(overflow_list[:,-1])
        wh_minmax = args[np.argmin(overflow_list[:,0][args])]

        return list(keys)[wh_minmax]

    def compare_value_minmax(new_values, old_values):
        if new_values[0] == old_values[0]:
            if new_values[1] < old_values[1]:
                return new_values
            else:
                return old_values
        elif new_values[0] < old_values[0]:
            return new_values
        else:
            return old_values

    for old_key in sorted(overflow_map.keys(), reverse=True):
        if old_key <= max_markers:
            break

        if len(overflow_map[old_key]):
            new_key = get_key_minmax(overflow_map[old_key].keys(), (overflow_map[old_key].values()))

            key_map[old_key] = new_key

            for key in overflow_map[old_key]:
                if max_markers < key < new_key: # add to new key
                    if key in overflow_map[new_key]:
                        overflow_map[new_key][key] = compare_value_minmax(overflow_map[new_key][key],
                                                                          overflow_map[old_key][key])
                    else:
                        overflow_map[new_key][key] = overflow_map[old_key][key]


                elif key > new_key and key > max_markers: # add new key to exisiting key
                    if new_key in overflow_map[key]:
                        overflow_map[key][new_key] = compare_value_minmax(overflow_map[key][new_key],
                                                                          overflow_map[old_key][key])
                    else:
                        overflow_map[key][new_key] = overflow_map[old_key][key]
        else:
            key_map[old_key] = 0

    for i in range(max_iter):
        if key_map.max() <= max_markers:
            break
        key_map = key_map[key_map]
    else:
        key_map[key_map > max_markers] = 0
    return key_map[fill]
