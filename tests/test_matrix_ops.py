import numpy as np
import pytest

from matrix_ops import (adjust_brightness, flip_horizontal, run_pipeline,
                        to_black_white, to_grayscale)


@pytest.fixture
def m():
    # 2 x 2 image: red, green / blue, near-white
    return np.array([[[255, 0, 0], [0, 255, 0]],
                     [[0, 0, 255], [250, 250, 250]]], dtype=np.uint8)


def test_grayscale_shape_and_values(m):
    g = to_grayscale(m)
    assert g.shape == (2, 2)
    assert g.tolist() == [[76, 150], [29, 250]]


def test_brightness_clips_instead_of_wrapping(m):
    assert adjust_brightness(m, 20)[1, 1].tolist() == [255, 255, 255]   # 250 + 20 -> 255, not 14
    assert adjust_brightness(m, -100)[1, 1].tolist() == [150, 150, 150]
    assert adjust_brightness(m, -300).max() == 0


def test_flip_reverses_columns(m):
    f = flip_horizontal(m)
    assert f[0, 0].tolist() == m[0, 1].tolist()
    assert f[0, 1].tolist() == m[0, 0].tolist()
    assert np.array_equal(flip_horizontal(f), m)                        # flipping twice = original


def test_black_white_only_0_or_255(m):
    bw = to_black_white(m, 128)
    assert set(np.unique(bw).tolist()) <= {0, 255}
    assert bw.tolist() == [[0, 255], [0, 255]]


def test_pipeline_combines_operations(m):
    out = run_pipeline(m, brightness=10, bw=True, flip=True, thresh=100)
    assert out.shape == (2, 2)
    assert set(np.unique(out).tolist()) <= {0, 255}


def test_pipeline_without_operations_returns_same_values(m):
    assert np.array_equal(run_pipeline(m), m)
