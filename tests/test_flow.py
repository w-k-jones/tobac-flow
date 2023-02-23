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

# Test smooth_flow
def test_smooth_flow_step_zero() -> None:
    """
    Test that when given flow arrays of all zero, the result is still zero
    """
    zero_flow = np.zeros([3,5,2], dtype=np.float32)
    assert np.all(np.stack(
        list(flow.smooth_flow_step(zero_flow, zero_flow))
    ) == 0)

def test_smooth_flow_step_one() -> None:
    """
    Test that for two flow arrays of value one the results are all one
    """
    one_flow = np.ones([3,5,2], dtype=np.float32)

    flow_forward, flow_backward = flow.smooth_flow_step(one_flow, -one_flow)
    assert np.all(flow_forward == 1)
    assert np.all(flow_backward == -1)

def test_smooth_flow_one_zero() -> None:
    """
    Test that when given one flow vector of 1 values, and one of 0, return 0.5 for valid warp locations
    """
    zero_flow = np.zeros([3,5,2], dtype=np.float32)
    one_flow = np.ones([3,5,2], dtype=np.float32)
    
    flow_forward, flow_backward = flow.smooth_flow_step(one_flow, zero_flow)

    assert np.all(flow_forward[:1,:3] == 0.5)
    assert np.all(flow_backward[:2,:4] == -0.5)

# Test calculate_flow_frame
def test_calculate_flow_frame_zero() -> None:
    """
    Test that when given a blob in the same position the result is flow vectors 
        of approximately zero
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    test_flow = flow.calculate_flow_frame(
        blob, blob, flow.select_of_model("DIS")
    )

    assert np.allclose(test_flow[0][...,0], 0, atol=0.05)
    assert np.allclose(test_flow[0][...,1], 0, atol=0.05)
    assert np.allclose(test_flow[1][...,0], 0, atol=0.05)
    assert np.allclose(test_flow[1][...,1], 0, atol=0.05)

def test_calculate_flow_frame_one_x() -> None:
    """
    Test that when given a blob offset by one in the x direction that the 
        resulting flow vectors are approximately one in the x axis
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    test_flow = flow.calculate_flow_frame(
        blob, np.roll(blob, 1, 1), flow.select_of_model("DIS")
    )

    assert np.allclose(test_flow[0][...,0], 1, atol=0.05)
    assert np.allclose(test_flow[0][...,1], 0, atol=0.05)
    assert np.allclose(test_flow[1][...,0], -1, atol=0.05)
    assert np.allclose(test_flow[1][...,1], 0, atol=0.05)

def test_calculate_flow_frame_one_y() -> None:
    """
    Test that when given a blob offset by one in the y direction that the 
        resulting flow vectors are approximately one in the y axis
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    test_flow = flow.calculate_flow_frame(
        blob, np.roll(blob, 1, 0), flow.select_of_model("DIS")
    )

    assert np.allclose(test_flow[0][...,0], 0, atol=0.05)
    assert np.allclose(test_flow[0][...,1], 1, atol=0.05)
    assert np.allclose(test_flow[1][...,0], 0, atol=0.05)
    assert np.allclose(test_flow[1][...,1], -1, atol=0.05)

def test_calculate_flow_frame_one_xy() -> None:
    """
    Test that when given a blob offset by minus one in the x and y direction 
        that the resulting flow vectors are approximately minus one in the x 
        and y axes
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    test_flow = flow.calculate_flow_frame(
        blob, np.roll(blob, -1, [0,1]), flow.select_of_model("DIS")
    )

    assert np.allclose(np.round(test_flow[0][...,0]), -1, atol=0.05)
    assert np.allclose(np.round(test_flow[0][...,1]), -1, atol=0.05)
    assert np.allclose(np.round(test_flow[1][...,0]), 1, atol=0.05)
    assert np.allclose(np.round(test_flow[1][...,1]), 1, atol=0.05)

def test_calculate_flow_frame_vr() -> None:
    """
    Test that variational refinement works
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    test_flow = flow.calculate_flow_frame(
        blob, np.roll(blob, -1, [0,1]), flow.select_of_model("DIS"), vr_steps=1
    )

    assert np.allclose(np.round(test_flow[0][...,0]), -1, atol=0.05)
    assert np.allclose(np.round(test_flow[0][...,1]), -1, atol=0.05)
    assert np.allclose(np.round(test_flow[1][...,0]), 1, atol=0.05)
    assert np.allclose(np.round(test_flow[1][...,1]), 1, atol=0.05)

def test_calculate_flow_frame_smoothing() -> None:
    """
    Test that smoothing works
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    test_flow = flow.calculate_flow_frame(
        blob, np.roll(blob, -1, [0,1]), flow.select_of_model("DIS"), 
        smoothing_steps=1
    )

    assert np.allclose(np.round(test_flow[0][...,0]), -1, atol=0.05)
    assert np.allclose(np.round(test_flow[0][...,1]), -1, atol=0.05)
    assert np.allclose(np.round(test_flow[1][...,0]), 1, atol=0.05)
    assert np.allclose(np.round(test_flow[1][...,1]), 1, atol=0.05)

# Test calculate flow
def test_calculate_flow_zero() -> None:
    """
    Test that calculate flow, given a stack of a blob in the same location, 
        returns zero for flow vectors
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    blob_stack = np.stack([blob]*3, 0)
    test_flow = flow.calculate_flow(blob_stack, "DIS")
    assert np.allclose(test_flow, 0, 0.05)

def test_calculate_flow_offset_one() -> None:
    """
    Test that calculate flow, given a stack of a blob offset by one in x and y 
        at each step, returns flow vectors of approximately one
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    blob_stack = np.stack(
        [np.roll(blob, -1, (0,1)), blob, np.roll(blob, 1, (0,1))]
    )
    test_flow = flow.calculate_flow(blob_stack, "DIS")
    assert np.allclose(np.around(test_flow[0]), 1, 0.05)
    assert np.allclose(np.around(test_flow[1]), -1, 0.05)

def test_calculate_flow_vr_steps() -> None:
    """
    Test that vr_steps work
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    blob_stack = np.stack(
        [np.roll(blob, -1, (0,1)), blob, np.roll(blob, 1, (0,1))]
    )
    test_flow = flow.calculate_flow(blob_stack, "DIS", vr_steps=1)
    assert np.allclose(np.around(test_flow[0]), 1, 0.05)
    assert np.allclose(np.around(test_flow[1]), -1, 0.05)

def test_calculate_flow_smoothing_passes() -> None:
    """
    Test that smoothing_passes work
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    blob_stack = np.stack(
        [np.roll(blob, -1, (0,1)), blob, np.roll(blob, 1, (0,1))]
    )
    test_flow = flow.calculate_flow(blob_stack, "DIS", smoothing_passes=1)
    assert np.allclose(np.around(test_flow[0]), 1, 0.05)
    assert np.allclose(np.around(test_flow[1]), -1, 0.05)

# Test create_flow
def test_create_flow() -> None:
    """
    Test that create flow returns an object with identical flow vectors to the 
        reult from calculate_flow
    """
    xx, yy = np.meshgrid(np.arange(15), np.arange(10))
    blob = flow.to_8bit((7**2-(xx-7)**2) * (4.5**2-(yy-4.5)**2))
    blob_stack = np.stack(
        [np.roll(blob, -1, (0,1)), blob, np.roll(blob, 1, (0,1))]
    )
    test_flow = flow.calculate_flow(blob_stack, "DIS")
    test_flow_obj = flow.create_flow(blob_stack, "DIS")
    assert isinstance(test_flow_obj, flow.Flow)
    assert np.allclose(test_flow[0], test_flow_obj.forward_flow, atol=0.05)
    assert np.allclose(test_flow[1], test_flow_obj.backward_flow, atol=0.05)

# Test Flow
def test_flow_init() -> None:
    """
    Test that we can successfully initiate a flow object
    """
    zeros = np.zeros([3,5,2])
    flow_obj = flow.Flow(zeros, zeros)
    assert isinstance(flow_obj, flow.Flow)
    assert flow_obj.shape == (3,5)

def test_flow_init_shape_mismatch() -> None:
    """
    Test that if the shapes of the optical flow vectors arrays are different an 
        exception is raised
    """
    zeros1 = np.zeros([3,5,2])
    zeros2 = np.zeros([2,4,2])
    with pytest.raises(ValueError):
        flow_obj = flow.Flow(zeros1, zeros2)

def test_flow_init_trailing_dimension_size_error() -> None:
    """
    Test that if the shapes of the optical flow vectors arrays have a size other 
        than 2 in the trailing dimension an exception is raised
    """
    zeros = np.zeros([3,5,1])
    with pytest.raises(ValueError):
        flow_obj = flow.Flow(zeros, zeros)

def test_flow_get_flow() -> None:
    """
    Test accessing flow property
    """
    zeros = np.zeros([3,5,2])
    flow_obj = flow.Flow(zeros, zeros)
    flow_vectors = flow_obj.flow
    assert np.all(flow_vectors[0] == flow_obj.forward_flow)
    assert np.all(flow_vectors[1] == flow_obj.backward_flow)

def test_flow_getitem() -> None:
    """
    Test __getitem__ dunder
    """
    zeros = np.zeros([3,5,2])
    flow_obj = flow.Flow(zeros, zeros)
    assert flow_obj[:2,:4].shape == (2,4)

