import numpy as np
import cv2

border_modes = {
    "constant": cv2.BORDER_CONSTANT,
    "nearest": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "mirror": cv2.BORDER_REFLECT_101,
    "wrap": cv2.BORDER_WRAP,
    "isolated": cv2.BORDER_ISOLATED,
    "transparent": cv2.BORDER_TRANSPARENT,
}

def select_border_mode(mode: str):
    if mode not in border_modes:
        raise ValueError("Invalid border mode")
    else:
        return border_modes[mode]

interp_modes = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

def select_interp_mode(mode: str):
    if mode not in interp_modes:
        raise ValueError("Invalid border mode")
    else:
        return interp_modes[mode]
    

def select_of_model(model: str) -> cv2.DenseOpticalFlow:
    """
    Initiates an opencv optical flow model

    Parameters
    ----------
    model : string
        The model to initatite. Must be one of 'Farneback', 'DeepFlow', 'PCA',
            'SimpleFlow', 'SparseToDense', 'DIS', 'DenseRLOF', 'DualTVL1'

    Returns
    -------
    of_model : cv2.DenseOpticalFlow
        opencv optical flow model
    """
    if model == "Farneback":
        of_model = cv2.optflow.createOptFlow_Farneback()
    elif model == "DeepFlow":
        of_model = cv2.optflow.createOptFlow_DeepFlow()
    elif model == "PCA":
        of_model = cv2.optflow.createOptFlow_PCAFlow()
    elif model == "SimpleFlow":
        of_model = cv2.optflow.createOptFlow_SimpleFlow()
    elif model == "SparseToDense":
        of_model = cv2.optflow.createOptFlow_SparseToDense()
    elif model == "DIS":
        of_model = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        of_model.setUseSpatialPropagation(True)
    elif model == "DenseRLOF":
        of_model = cv2.optflow.createOptFlow_DenseRLOF()
        raise NotImplementedError(
            "DenseRLOF requires multi-channel input which is currently not implemented"
        )
    elif model == "DualTVL1":
        of_model = cv2.optflow.createOptFlow_DualTVL1()
    else:
        raise ValueError(
            "'model' parameter must be one of: 'Farneback', 'DeepFlow', 'PCA', 'SimpleFlow', 'SparseToDense', 'DIS', 'DenseRLOF', 'DualTVL1'"
        )

    return of_model

def warp_flow(img: np.ndarray, flow: np.ndarray, method: str = "linear", border: str = "constant") -> np.ndarray:
    """
    Warp an image according to a given set of optical flow vectors
    """
    h, w = flow.shape[:2]
    locs = flow.copy()
    locs[:, :, 0] += np.arange(w)
    locs[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, locs, None, select_interp_mode(method), None, select_border_mode(border), np.nan)
    return res


__all__ = (
    "select_border_mode",
    "select_interp_mode",
    "select_of_model",
    "warp_flow",
)