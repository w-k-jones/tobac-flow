import numpy as np
import cv2
import pytest
from tobac_flow import flow

# Test of_model
def test_vr_model() -> None:
    assert isinstance(
        flow.vr_model, cv2.VariationalRefinement
    ), "vr_model is not an instance of cv2.VariationalRefinement"

def test_select_of_model_Farneback() -> None:
    assert isinstance(flow.select_of_model("Farneback"), cv2.DenseOpticalFlow)

def test_select_of_model_DeepFlow() -> None:
    assert isinstance(flow.select_of_model("DeepFlow"), cv2.DenseOpticalFlow)

def test_select_of_model_PCA() -> None:
    assert isinstance(flow.select_of_model("PCA"), cv2.DenseOpticalFlow)

def test_select_of_model_SimpleFlow() -> None:
    assert isinstance(flow.select_of_model("SimpleFlow"), cv2.DenseOpticalFlow)

def test_select_of_model_SparseToDense() -> None:
    assert isinstance(
        flow.select_of_model("SparseToDense"), cv2.DenseOpticalFlow
    )

def test_select_of_model_DIS() -> None:
    assert isinstance(flow.select_of_model("DIS"), cv2.DISOpticalFlow)

def test_select_of_model_DualTVL1() -> None:
    assert isinstance(
        flow.select_of_model("DualTVL1"), cv2.optflow.DualTVL1OpticalFlow
    )

def test_select_of_model_DenseRLOF() -> None:
    with pytest.raises(NotImplementedError):
        flow.select_of_model("DenseRLOF")

def test_select_of_model_invalid_model() -> None:
    with pytest.raises(ValueError):
        flow.select_of_model("not_an_of_model")

# Test to_8bit
def test_to_8bit_zeros() -> None:
    """
    Test that an array of all zeros returns all zeros
    """
    arr = np.zeros(5)
    assert np.all(flow.to_8bit(arr)==0)

def test_to_8bit_ones() -> None:
    """
    Test that an array of all ones returns all zeros if no vmin/vmax set
    """
    arr = np.ones(5)
    assert np.all(flow.to_8bit(arr)==0)

def test_to_8bit_ones_vmax() -> None:
    """
    Test that an array of all ones returns all 255 if vmin/vmax set to 0/1
    """
    arr = np.ones(5)
    assert np.all(flow.to_8bit(arr, vmin=0, vmax=1)==255)

def test_to_8bit_arange() -> None:
    """
    Test that an array of integers from 0 to 255 returns the same values
    """
    arr = np.arange(256)
    assert np.all(flow.to_8bit(arr)==arr)

def test_to_8bit_arange_vmin_vmax() -> None:
    """
    Test that an array of integers from 10 to 265 returns the values in the
        range 0-255
    """
    arr = np.arange(256)
    assert np.all(flow.to_8bit(arr + 10, vmin=10, vmax=10+255)==arr)

def test_warp_flow_zero_flow() -> None:
    """
    Test that zero flow vectors returns the same image
    """
    test_arr = np.arange(15, dtype=np.float32).reshape(3,5)
    flow_arr = np.zeros(test_arr.shape+(2,), dtype=np.float32)
    warp_arr = flow.warp_flow(test_arr, flow_arr)
    # Remove NaN locations where out of frame
    wh_nan = np.isnan(warp_arr)
    assert np.all(warp_arr[~wh_nan]==test_arr[~wh_nan])

def test_warp_flow_one_x_flow() -> None:
    """
    Test that size 1 flow vectors in the x dimension returns the same image
        shifted 1 in x
    """
    test_arr = np.arange(15, dtype=np.float32).reshape(3,5)
    flow_arr = np.zeros(test_arr.shape+(2,), dtype=np.float32)
    flow_arr[...,0] = 1
    warp_arr = flow.warp_flow(test_arr, flow_arr)[:,:-1]
    # Remove NaN locations where out of frame
    wh_nan = np.isnan(warp_arr)
    assert np.all(warp_arr[~wh_nan]==test_arr[:,1:][~wh_nan])

def test_warp_flow_one_y_flow() -> None:
    """
    Test that size 1 flow vectors in the y dimension returns the same image
        shifted 1 in y
    """
    test_arr = np.arange(15, dtype=np.float32).reshape(3,5)
    flow_arr = np.zeros(test_arr.shape+(2,), dtype=np.float32)
    flow_arr[...,1] = 1
    warp_arr = flow.warp_flow(test_arr, flow_arr)[:-1]
    # Remove NaN locations where out of frame
    wh_nan = np.isnan(warp_arr)
    assert np.all(warp_arr[~wh_nan]==test_arr[1:][~wh_nan])

def test_warp_flow_one_xy_flow() -> None:
    """
    Test that size 1 flow vectors in the x and y dimensions returns the same
        image shifted 1 in x and y
    """
    test_arr = np.arange(15, dtype=np.float32).reshape(3,5)
    flow_arr = np.zeros(test_arr.shape+(2,), dtype=np.float32)
    flow_arr[...,:] = 1
    warp_arr = flow.warp_flow(test_arr, flow_arr)[:-1,:-1]
    # Remove NaN locations where out of frame
    wh_nan = np.isnan(warp_arr)
    assert np.all(warp_arr[~wh_nan]==test_arr[1:,1:][~wh_nan])

def test_warp_flow_half_x_flow() -> None:
    """
    Test that size 0.5 flow vectors in the x dimension returns a linear
        combination of the input image shifted 0.5 in x
    """
    test_arr = np.arange(15, dtype=np.float32).reshape(3,5)
    flow_arr = np.zeros(test_arr.shape+(2,), dtype=np.float32)
    flow_arr[...,0] = 0.5
    warp_arr = flow.warp_flow(test_arr, flow_arr)[:,:-1]
    # Remove NaN locations where out of frame
    wh_nan = np.isnan(warp_arr)
    assert np.all(
        warp_arr[~wh_nan] == ((test_arr[:,1:] + test_arr[:,:-1])[~wh_nan] * 0.5)
    )
