import numpy as np
from numpy import ma
import pandas as pd
import xarray as xr
import cv2 as cv
from scipy import ndimage as ndi

class Flow:
    """
    Class to perform semi-lagrangian operations using optical flow
    """
    def __init__(self, dataset, smoothing_passes=1, flow_kwargs={}):
        self.get_flow(dataset, smoothing_passes, flow_kwargs)

    def get_flow(self, data, smoothing_passes, flow_kwargs):
        self.shape = data.shape
        self.flow_for = np.full(self.shape+(2,), np.nan, dtype=np.float32)
        self.flow_back = np.full(self.shape+(2,), np.nan, dtype=np.float32)

        for i in range(self.shape[0]-1):
            print(i, end='\r')
            a, b = data[i].compute().data, data[i+1].compute().data

            self.flow_for[i] = self.cv_flow(a, b, **flow_kwargs)
            self.flow_back[i+1] = self.cv_flow(b, a, **flow_kwargs)
            if smoothing_passes > 0:
                for j in range(smoothing_passes):
                    self._smooth_flow_step(i)

        self.flow_back[0] = -self.flow_for[0]
        self.flow_for[-1] = -self.flow_back[-1]

    def to_8bit(self, array, vmin=None, vmax=None):
        """
        Converts an array to an 8-bit range between 0 and 255
        """
        if vmin is None:
            vmin = np.nanmin(array)
        if vmax is None:
            vmax = np.nanmax(array)
        array_out = (array-vmin) * 255 / (vmax-vmin)
        return array_out.astype('uint8')

    def cv_flow(self, a, b, pyr_scale=0.5, levels=5, winsize=16, iterations=3,
                poly_n=5, poly_sigma=1.1, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN):
        """
        Wrapper function for cv.calcOpticalFlowFarneback
        """
        flow = cv.calcOpticalFlowFarneback(self.to_8bit(a), self.to_8bit(b), None,
                                           pyr_scale, levels, winsize, iterations,
                                           poly_n, poly_sigma, flags)
        return flow

    def _warp_flow_step(self, img, step, method='linear', direction='forward',
                        stencil=ndi.generate_binary_structure(3,1)[0]):
        if img.shape != self.shape[1:]:
            raise ValueError("Image shape does not match flow shape")
        if method == 'linear':
            method = cv.INTER_LINEAR
        elif method =='nearest':
            method = cv.INTER_NEAREST
        else:
            raise ValueError("method must be either 'linear' or 'nearest'")

        h, w = self.shape[1:]
        n = np.sum(stencil!=0)
        offsets = np.stack(np.where(stencil!=0), -1)[:,np.newaxis,np.newaxis,:]-1
        locations = np.tile(offsets, [1,h,w,1]).astype(np.float32)
        locations += np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1)
        if direction=='forward':
            locations += self.flow_for[step]
        elif direction=='backward':
            locations += self.flow_back[step]

        out_img = np.full([n*h,w,2], np.nan)

        return cv.remap(img, locations.reshape([n*h,w,2]), None,
                        method, out_img, cv.BORDER_CONSTANT, np.nan).reshape([n,h,w])

    def _smooth_flow_step(self, step):
        flow_for_warp = np.full_like(self.flow_for[step], np.nan)
        flow_back_warp = np.full_like(self.flow_back[step+1], np.nan)

        flow_for_warp[...,0] = -self._warp_flow_step(self.flow_back[step+1,...,0], step)
        flow_for_warp[...,1] = -self._warp_flow_step(self.flow_back[step+1,...,1], step)
        flow_back_warp[...,0] = -self._warp_flow_step(self.flow_for[step,...,0], step+1, direction='backward')
        flow_back_warp[...,1] = -self._warp_flow_step(self.flow_for[step,...,1], step+1, direction='backward')

        self.flow_for[step] = np.nanmean([self.flow_for[step],
                                          flow_for_warp], 0)
        self.flow_back[step+1] = np.nanmean([self.flow_back[step+1],
                                             flow_back_warp], 0)

    def convolve(self, data, structure=ndi.generate_binary_structure(3,1), func=None,
                 method='linear', dtype=np.float32):
        assert structure.shape == (3,3,3), "Structure input must be a 3x3x3 array"
        assert data.shape == self.shape, "Data input must have the same shape as the Flow object"
        n_structure = np.count_nonzero(structure)
        wh_layer = np.nonzero(structure)
        struct_factor = structure[np.nonzero(structure)]
        if func is None:
            out_array = np.full((n_structure,)+self.shape, np.nan, dtype=dtype)
        else:
            out_array = np.full(self.shape, np.nan, dtype=dtype)
        img_step = -1

        for step in range(data.shape[0]):
    #       Construct temporary array for the data from this time step
            temp = np.full((n_structure,)+data.shape[1:], np.nan)

    #       Now loop through elements of structure


    #           For backward steps:
            n_back = np.count_nonzero(structure[0])
            if n_back > 0 and step > 0:
                if img_step != step-1:
                    if hasattr(data, 'compute'):
                        img = data[step-1].compute().data
                    else:
                        img = data[step-1]
                    img_step = step-1
                temp[:n_back] = self._warp_flow_step(img, step,
                                                     method=method,
                                                     direction='backward',
                                                     stencil=structure[0]) \
                                * struct_factor[:n_back,np.newaxis,np.newaxis]
    #           For forward steps:
            n_forward = np.count_nonzero(structure[2])
            if n_forward > 0 and step < data.shape[0]-1:
                if img_step != step+1:
                    if hasattr(data, 'compute'):
                        img = data[step+1].compute().data
                    else:
                        img = data[step+1]
                    img_step = step+1
                temp[-n_forward:] = self._warp_flow_step(img, step,
                                                         method=method,
                                                         direction='forward',
                                                         stencil=structure[0]) \
                                    * struct_factor[-n_forward:,np.newaxis,np.newaxis]
    #           For same time step:
            for i in range(n_back, n_structure-n_forward):
                if img_step != step:
                    if hasattr(data, 'compute'):
                        img = data[step].compute().data
                    else:
                        img = data[step]
                    img_step = step
                if wh_layer[1][i]==1 and wh_layer[2][i]==1:
                    temp[i] = img * struct_factor[i]
                else:
                    temp[i,
                         (1 if wh_layer[2][i]==0 else 0):(-1 if wh_layer[2][i]==2 else None),
                         (1 if wh_layer[1][i]==0 else 0):(-1 if wh_layer[1][i]==2 else None)] \
                        = img[(1 if wh_layer[2][i]==2 else 0):(-1 if wh_layer[2][i]==0 else None),
                              (1 if wh_layer[1][i]==2 else 0):(-1 if wh_layer[1][i]==0 else None)] \
                          * struct_factor[i]

            if func is None:
                out_array[:,step] = temp
            else:
                out_array[step] = func(temp)
        # Propagate nan locations forward
        if not func is None:
            out_array[~np.isfinite(data)] = np.nan
        return out_array

    def diff(self, data, dtype=np.float32):
        diff_struct = np.zeros([3,3,3])
        diff_struct[:,1,1] = 1
        diff = self.convolve(data, structure=diff_struct,
                             func=lambda x:np.nansum([x[2]-x[1], x[1]-x[0]], axis=0) \
                                           * 1/np.maximum(np.sum([np.isfinite(x[2]), np.isfinite(x[0])], 0), 1))
        return diff

    def _sobel_matrix(self, ndims):
        sobel_matrix = np.array([-1,0,1])
        for i in range(ndims-1):
            sobel_matrix = np.multiply.outer(np.array([1,2,1]), sobel_matrix)
        return sobel_matrix


    def _sobel_func_uphill(self, x):
        sobel_matrix = self._sobel_matrix(3)
        x = np.fmax(x-x[13],0)
        out_array = np.nansum(x * sobel_matrix.ravel()[:,np.newaxis,np.newaxis], 0)**2
        out_array += np.nansum(x * sobel_matrix.transpose([1,2,0]).ravel()[:,np.newaxis,np.newaxis], 0)**2
        out_array += np.nansum(x * sobel_matrix.transpose([2,0,1]).ravel()[:,np.newaxis,np.newaxis], 0)**2

        return out_array ** 0.5

    def _sobel_func_downhill(self, x):
        sobel_matrix = self._sobel_matrix(3)
        x = np.fmin(x-x[13],0)
        out_array = np.nansum(x * sobel_matrix.ravel()[:,np.newaxis,np.newaxis], 0)**2
        out_array += np.nansum(x * sobel_matrix.transpose([1,2,0]).ravel()[:,np.newaxis,np.newaxis], 0)**2
        out_array += np.nansum(x * sobel_matrix.transpose([2,0,1]).ravel()[:,np.newaxis,np.newaxis], 0)**2

        return out_array ** 0.5

    def _sobel_func(self, x):
        sobel_matrix = self._sobel_matrix(3)
        x -= x[13]
        out_array = np.nansum(x * sobel_matrix.ravel()[:,np.newaxis,np.newaxis], 0)**2
        out_array += np.nansum(x * sobel_matrix.transpose([1,2,0]).ravel()[:,np.newaxis,np.newaxis], 0)**2
        out_array += np.nansum(x * sobel_matrix.transpose([2,0,1]).ravel()[:,np.newaxis,np.newaxis], 0)**2

        return out_array ** 0.5

    def sobel(self, data, method='linear', direction=None):
        if direction == 'uphill':
            return self.convolve(data, structure=np.ones((3,3,3)),
                                 func=self._sobel_func_uphill, method=method)
        if direction == 'downhill':
            return self.convolve(data, structure=np.ones((3,3,3)),
                                 func=self._sobel_func_downhill, method=method)
        else:
            return self.convolve(data, structure=np.ones((3,3,3)),
                                 func=self._sobel_func, method=method)

    def watershed(self, field, markers, mask=None,
                  structure=ndi.generate_binary_structure(3,1),
                  max_iter=100, debug_mode=False):
        from .legacy_flow import Flow_Func, flow_network_watershed

        l_flow = Flow_Func(self.flow_for[...,0], self.flow_back[...,0],
                              self.flow_for[...,1], self.flow_back[...,1])

        return flow_network_watershed(field, markers, l_flow,
                                         mask=mask, structure=structure,
                                         max_iter=max_iter,
                                         debug_mode=debug_mode)

    def label(self, data, structure=ndi.generate_binary_structure(3,1),
              dtype=np.int32, overlap=0, subsegment_shrink=0):
        return flow_label(self, data, structure=structure, dtype=dtype,
                          overlap=overlap, subsegment_shrink=subsegment_shrink)

def subsegment_labels(input_mask, shrink_factor=0.1):
    from tobac_flow.analysis import flat_label
    from skimage.segmentation import watershed
    labels = flat_label(input_mask!=0)
    pixel_counts = np.bincount(labels.ravel())
    dist_mask = ndi.morphology.distance_transform_edt(labels, [1e9,1,1])/((pixel_counts/np.pi)**0.5)[labels]
    shrunk_labels = flat_label(dist_mask>shrink_factor)
    shrunk_markers = shrunk_labels.copy()
    shrunk_markers[labels==0] = -1
    struct = ndi.generate_binary_structure(3,1)
    struct[0] = 0
    struct[-1] = 0
    subseg_labels = np.zeros_like(labels)
    for i in range(subseg_labels.shape[0]):
        subseg_labels[i] = watershed(-dist_mask[i], shrunk_markers[i], mask=labels[i]!=0)

    return subseg_labels

# implement minimum overlap for flow_label function
def flow_label(flow, mask, structure=ndi.generate_binary_structure(3,1), dtype=np.int32, overlap=0, subsegment_shrink=0):
    """
    Label 3d connected objects in a semi-Lagrangian reference frame
    """
    from tobac_flow.analysis import flat_label
    from collections import deque
#     Get flat (2d) labels
    if subsegment_shrink == 0:
        flat_labels = flat_label(mask.astype(bool), structure=structure).astype(dtype)
    else:
        flat_labels = subsegment_labels(mask.astype(bool), shrink_factor=subsegment_shrink)

    back_labels, forward_labels = flow.convolve(flat_labels, method='nearest', dtype=dtype,
                                              structure=structure*np.array([1,0,1])[:,np.newaxis, np.newaxis])

    processed_labels = []
    label_map = {}

    bins = np.cumsum(np.bincount(flat_labels.ravel()))
    args = np.argsort(flat_labels.ravel())

    for label in range(1, bins.size):
        if label not in processed_labels:
            label_map[label] = [label]
            processed_labels.append(label)

            i = 0
            while i < len(label_map[label]):
                find_neighbour_labels(label_map[label][i], label_map[label], bins, args,
                                      processed_labels, forward_labels, back_labels,
                                      overlap=overlap)
                i+=1

    new_labels = np.zeros(mask.shape, dtype=dtype)

    for ik, k in enumerate(label_map):
        for i in label_map[k]:
            if bins[i]>bins[i-1]:
                new_labels.ravel()[args[bins[i-1]:bins[i]]] = ik+1

    assert np.all(new_labels.astype(bool)==mask.astype(bool))
    return new_labels

def find_neighbour_labels(label, label_stack, bins, args, processed_labels,
                          forward_labels, back_labels, overlap=0):
    """
    Find the neighbouring labels at the previous and next time steps to a given
    label
    """
    if bins[label]>bins[label-1]: #check that there are any pixels in this label
        forward_lap = forward_labels.ravel()[args[bins[label-1]:bins[label]]]
        forward_bins = np.bincount(np.maximum(forward_lap,0))
        for new_label in np.unique(forward_lap):
            if (new_label>0 and
                new_label not in processed_labels and
                forward_bins[new_label] >= overlap*np.minimum(bins[label]-bins[label-1], bins[new_label]-bins[new_label-1])):

                label_stack.append(new_label)
                processed_labels.append(new_label)

        backward_lap = back_labels.ravel()[args[bins[label-1]:bins[label]]]
        backward_bins = np.bincount(np.maximum(backward_lap,0))
        for new_label in np.unique(backward_lap):
            if (new_label>0 and
                new_label not in processed_labels and
                backward_bins[new_label] >= overlap*np.minimum(bins[label]-bins[label-1], bins[new_label]-bins[new_label-1])):

                label_stack.append(new_label)
                processed_labels.append(new_label)




#Old, recursive version. Replaced as in some cases could hit the recursion cap
# def find_neighbour_labels(label, bins, args, processed_labels, forward_labels, back_labels):
#     """
#     Recursive function to find all the neighbouring, overlapping 2d regions for each label
#     """
#     processed_labels.append(label)
#     if bins[label]>bins[label-1]:
#         return_list = [label]
#         for i in np.unique(forward_labels.ravel()[args[bins[label-1]:bins[label]]]):
#             if i>0 and i not in processed_labels:
#                 return_list.extend(find_neighbour_labels(i, bins, args, processed_labels,
#                                                          forward_labels, back_labels))
#         for i in np.unique(back_labels.ravel()[args[bins[label-1]:bins[label]]]):
#             if i>0 and i not in processed_labels:
#                 return_list.extend(find_neighbour_labels(i, bins, args, processed_labels,
#                                                          forward_labels, back_labels))
#
#     else:
#         return_list = []
#     return return_list


# class Flow_dev:
#     """
#     Class to perform semi-lagrangian operations using optical flow
#     """
#     def __init__(self, dataset, smoothing_passes=1, flow_kwargs={}):
#         get_flow(self, dataset, smoothing_passes, flow_kwargs)
#
#     def get_flow(self, data, smoothing_passes, flow_kwargs):
#         """
#         Get both forwards and backwards optical flow vectors along the time
#         dimension from an array with dimensions (time, y, x)
#         """
#         self.shape = self.shape
#         self.flow_for = np.full(self.shape+(2,), np.nan, dtype=np.float32)
#         self.flow_back = np.full(self.shape+(2,), np.nan, dtype=np.float32)
#
#         b = dataset[0].compute().data
#         for i in range(self.shape[0]-1):
#             a, b = b, dataset[i+1].compute().data
#             self.flow_for[i] = cv_flow(a, b, **flow_kwargs)
#             self.flow_back[i+1] = cv_flow(b, a, **flow_kwargs)
#             if smoothing_passes > 0:
#                 for j in range(smoothing_passes):
#                     self._smooth_flow_step(i)
#
#         self.flow_back[0] = -self.flow_for[0]
#         self.flow_for[-1] = -self.flow_back[-1]
#
#     def _smooth_flow_step(self, step):
#         flow_for_warp = np.full_like(self.flow_for[step], np.nan)
#         flow_back_warp = np.full_like(self.flow_back[step+1], np.nan)
#
#         flow_for_warp[...,0] = -self._warp_flow_step(self.flow_back[step+1,...,0], step)
#         flow_for_warp[...,1] = -self._warp_flow_step(self.flow_back[step+1,...,1], step)
#         flow_back_warp[...,0] = -self._warp_flow_step(self.flow_for[step,...,0], step+1, direction='backward')
#         flow_back_warp[...,1] = -self._warp_flow_step(self.flow_for[step,...,1], step+1, direction='backward')
#
#         self.flow_for[step] = np.nanmean([self.flow_for[step],
#                                           flow_for_warp], 0)
#         self.flow_back[step+1] = np.nanmean([self.flow_back[step+1],
#                                              flow_back_warp], 0)

def to_8bit(array, vmin=None, vmax=None):
    """
    Converts an array to an 8-bit range between 0 and 255 with dtype uint8
    """
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    array_out = (array-vmin) * 255 / (vmax-vmin)
    return array_out.astype('uint8')

"""
Dicts to convert keyword inputs to opencv flags for flow keywords
"""
flow_flags = {'default':0, 'gaussian':cv.OPTFLOW_FARNEBACK_GAUSSIAN}

def cv_flow(a, b, pyr_scale=0.5, levels=5, winsize=16, iterations=3,
            poly_n=5, poly_sigma=1.1, flags='gaussian'):
    """
    Wrapper function for cv.calcOpticalFlowFarneback
    """
    assert flags in flow_flags, \
        f"{flags} not a valid input for flags keyword, input must be one of {list(flow_flags.keys())}"

    flow = cv.calcOpticalFlowFarneback(to_8bit(a), to_8bit(b), None,
                                       pyr_scale, levels, winsize, iterations,
                                       poly_n, poly_sigma, flow_flags[flags])
    return flow

"""
Dicts to convert keyword inputs to opencv flags for remap keywords
"""
border_modes = {'constant':cv.BORDER_CONSTANT,
                'nearest':cv.BORDER_REPLICATE,
                'reflect':cv.BORDER_REFLECT,
                'mirror':cv.BORDER_REFLECT_101,
                'wrap':cv.BORDER_WRAP,
                'isolated':cv.BORDER_ISOLATED,
                'transparent':cv.BORDER_TRANSPARENT}

interp_modes = {'nearest':cv.INTER_NEAREST,
                'linear':cv.INTER_LINEAR,
                'cubic':cv.INTER_CUBIC,
                'lanczos':cv.INTER_LANCZOS4}

def cv_remap(img, locs, border_mode='constant', interp_mode='linear',
             cval=np.nan, dtype=None):
    """
    Wrapper function for cv.remap
    """
    assert border_mode in border_modes, \
        f"{border_mode} not a valid input for border_mode keyword, input must be one of {list(border_modes.keys())}"
    assert interp_mode in interp_modes, \
        f"{interp_mode} not a valid input for border_mode keyword, input must be one of {list(interp_modes.keys())}"

    if not dtype:
        dtype = img.dtype
    out_img = np.full(locs.shape[:-1], cval, dtype=dtype)

    cv.remap(img, locs.astype(np.float32), None, nterp_modes[interp_mode],
             out_img, border_modes[border_mode], cval)
    return out_img
