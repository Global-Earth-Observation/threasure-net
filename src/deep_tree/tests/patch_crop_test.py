
import os
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from deep_tree.datamodules.prepare_sample import RandomPatchCrop


def create_dummy_tiff(path: str, data: np.array, transform=None) -> None:
    """Helper to create a single-band GeoTIFF."""
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        transform=transform or from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def dummy_rasters(tmp_path: str) -> tuple[np.array, np.array, np.array, np.array]:
    """Create LiDAR (1000x1000) and S2 (150x150) rasters.

    LiDAR resolution = 1 m
    S2 resolution = 10 m
    cropped_s2_patch_size = 64  -> cropped_lidar_size = 640
    S2 margin = 25 (prepared)   -> S2 patch with margin = 64 + 50 = 114
    We make S2 raster 150x150 to safely contain that.
    """
    lidar_h = lidar_w = 1000
    s2_h = s2_w = 150

    lidar_data = np.arange(lidar_h * lidar_w, dtype=np.int16).reshape(lidar_h, lidar_w)
    s2_data = np.arange(s2_h * s2_w, dtype=np.int16).reshape(s2_h, s2_w)

    lidar_path = tmp_path / "lidar.tif"
    s2_path = tmp_path / "s2.tif"

    create_dummy_tiff(lidar_path, lidar_data)
    create_dummy_tiff(s2_path, s2_data)

    return str(lidar_path), str(s2_path), lidar_data, s2_data


def test_s2_margin_clipping(
        dummy_rasters: tuple[np.array, np.array, np.array, np.array]
) -> None:
    lidar_path, s2_path, lidar_data, s2_data = dummy_rasters

    # Patch sizes
    cropped_s2_size = 64
    lidar_patch_size = 1000
    margin_s2 = 8  # test with margin smaller than 25

    rpc = RandomPatchCrop(
        lidar_folder=os.path.dirname(lidar_path),
        lidar_mask_folder=None,
        s2_folder=os.path.dirname(s2_path),
        cropped_s2_patch_size=cropped_s2_size,
        lidar_patch_size=lidar_patch_size,
        margin_s2=margin_s2,
        training=False,  # force central crop
    )

    # Compute central coords
    x_c, y_c = rpc.get_random_coords()
    x_c_s2, y_c_s2 = int(x_c / rpc.scale), int(y_c / rpc.scale)

    lidar_crop, _ = rpc.get_lidar(lidar_path, x_c, y_c)
    s2_crop, _ = rpc.get_s2(s2_path, x_c_s2, y_c_s2)

    # Expected window offsets for S2
    expected_x = x_c_s2 + 25 - margin_s2
    expected_y = y_c_s2 + 25 - margin_s2
    expected_size = cropped_s2_size + 2 * margin_s2

    assert s2_crop.shape[1] == expected_size
    assert s2_crop.shape[2] == expected_size

    # Check that the clipped values correspond to the correct part of the raster
    # (top-left pixel value should match s2_data at expected_y, expected_x)
    expected_value = s2_data[expected_y, expected_x]
    assert int(s2_crop[0, 0, 0]) == int(expected_value)


@pytest.mark.parametrize("margin", [0, 16, 25, 30])
def test_s2_different_margins(
        dummy_rasters: tuple[np.array, np.array, np.array, np.array],
        margin: int
) -> None:
    lidar_path, s2_path, lidar_data, s2_data = dummy_rasters

    rpc = RandomPatchCrop(
        lidar_folder=os.path.dirname(lidar_path),
        lidar_mask_folder=None,
        s2_folder=os.path.dirname(s2_path),
        cropped_s2_patch_size=64,
        lidar_patch_size=1000,
        margin_s2=margin,
        training=False,
    )

    x_c, y_c = rpc.get_random_coords()
    x_c_s2, y_c_s2 = int(x_c / rpc.scale), int(y_c / rpc.scale)

    s2_crop, _ = rpc.get_s2(s2_path, x_c_s2, y_c_s2)

    # Expected window offsets for S2
    expected_x = x_c_s2 + 25 - margin
    expected_y = y_c_s2 + 25 - margin

    expected_size = rpc.cropped_s2_size + 2 * margin
    assert s2_crop.shape[1] == expected_size
    assert s2_crop.shape[2] == expected_size

    # Check that the clipped values correspond to the correct part of the raster
    # (top-left pixel value should match s2_data at expected_y, expected_x)
    expected_value = s2_data[expected_y, expected_x]
    assert int(s2_crop[0, 0, 0]) == int(expected_value)


def test_random_coords_with_seed(dummy_rasters):
    """Random crop reproducible via RNG seed."""
    np.random.seed(42)
    lidar_path, s2_path, _, _ = dummy_rasters

    rpc = RandomPatchCrop(
        lidar_folder=os.path.dirname(lidar_path),
        lidar_mask_folder=None,
        s2_folder=os.path.dirname(s2_path),
        cropped_s2_patch_size=64,
        lidar_patch_size=1000,
        margin_s2=16,
        training=True,  # random crop
    )

    x_c, y_c = rpc.get_random_coords()
    print(x_c, y_c)
    # Stable across runs due to seed
    assert x_c.is_integer()
    assert y_c.is_integer()
    assert 0 <= x_c <= (rpc.lidar_size - rpc.cropped_lidar_size)
    assert 0 <= y_c <= (rpc.lidar_size - rpc.cropped_lidar_size)


def test_random_coords_with_mock(monkeypatch, dummy_rasters):
    """Random crop controlled via monkeypatch of np.random.choice."""
    lidar_path, s2_path, _, _ = dummy_rasters

    def fake_choice(arr, size, replace=True):
        return np.array([100, 200])  # force coords

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    rpc = RandomPatchCrop(
        lidar_folder=os.path.dirname(lidar_path),
        lidar_mask_folder=None,
        s2_folder=os.path.dirname(s2_path),
        cropped_s2_patch_size=64,
        lidar_patch_size=1000,
        margin_s2=16,
        training=True,
    )

    x_c, y_c = rpc.get_random_coords()
    assert (x_c, y_c) == (100, 200)
