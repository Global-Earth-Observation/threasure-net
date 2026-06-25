#!/usr/bin/env python
# coding: utf-8

"""
Sun and viewing angle processor.
We reconstruct angles from *MTD.aux file
Some tips are taken from SNAP forum:
https://forum.step.esa.int/t/generate-view-angles-from-metadata-sentinel-2/5598/2
"""

import os
import sys

import numpy as np
import xml.etree.ElementTree as ET
from collections import namedtuple, defaultdict

import pandas as pd
from scipy.interpolate import RegularGridInterpolator, griddata
from pyproj import CRS, Transformer
from scipy.ndimage import generic_filter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=sys.maxsize)


Angles = namedtuple('Angles', 's_z s_a v_z v_a')

path_metadata = "/your/path/input_data/metadata"


def get_zun_zenith(root: ET.Element) -> np.array:
    '''Extract sun zenith values'''
    sun_zenith_values = []
    for zenith in root.findall('.//Sun_Angles_Grids/Zenith/Values_List/VALUES'):
        row = [float(v) if v != 'NaN' else np.nan for v in zenith.text.split()]
        sun_zenith_values.append(row)

    # Convert to numpy array for easier handling
    return upscale(np.array(sun_zenith_values)[:22, :22])


def get_zun_azimuth(root: ET.Element) -> np.array:
    '''Extract sun azimuth  values'''
    sun_azimuth_values = []
    for azimuth in root.findall('.//Sun_Angles_Grids/Azimuth/Values_List/VALUES'):
        row = [float(v) if v != 'NaN' else np.nan for v in azimuth.text.split()]
        sun_azimuth_values.append(row)

    # Convert to numpy array for easier handling
    return upscale(np.array(sun_azimuth_values)[:22, :22])


def fill_nans_axiswise(arr):
    filled = arr.copy()
    for i in range(arr.shape[0]):
        series = pd.Series(arr[i, :])
        filled[i, :] = series.interpolate(method="nearest", limit_direction="both").to_numpy()
    return filled


# def fill_nans_axiswise(arr,
#                        axis=1,
#                        method="linear",
#                        fill_all_nan=None):
#     """
#     Fill NaNs along rows (axis=1) or columns (axis=0) using pandas interpolation.
#     - arr: 2D numpy array
#     - axis: axis along which to treat each 1D series (1 = rows, 0 = columns)
#     - method: interpolation method supported by pandas.Series.interpolate (e.g. "linear", "nearest")
#     - fill_all_nan: value to use if an entire row/col is NaN (None => leave as NaN)
#     Returns a float64 array with NaNs filled where possible.
#     """
#     if arr.ndim != 2:
#         raise ValueError("This helper expects a 2D array (bands excluded).")
#
#     a = arr.astype("float64", copy=True)  # make sure we can store NaNs / floats
#     out = np.empty_like(a)
#
#     if axis == 1:
#         outer = a.shape[0]
#         length = a.shape[1]
#         for i in range(outer):
#             s = pd.Series(a[i, :])
#             if s.isna().all():
#                 # nothing to interpolate — either fill with provided value or keep NaNs
#                 if fill_all_nan is not None:
#                     out[i, :] = fill_all_nan
#                 else:
#                     out[i, :] = s.to_numpy()
#                 continue
#
#             # interpolate, allow filling at edges by doing ffill/bfill afterwards
#             s = s.interpolate(method=method, limit_direction="both")
#             s = s.fillna(method="ffill").fillna(method="bfill")
#             out[i, :] = s.to_numpy()
#
#     elif axis == 0:
#         outer = a.shape[1]
#         length = a.shape[0]
#         for j in range(outer):
#             s = pd.Series(a[:, j])
#             if s.isna().all():
#                 if fill_all_nan is not None:
#                     out[:, j] = fill_all_nan
#                 else:
#                     out[:, j] = s.to_numpy()
#                 continue
#
#             s = s.interpolate(method=method, limit_direction="both")
#             s = s.fillna(method="ffill").fillna(method="bfill")
#             out[:, j] = s.to_numpy()
#     else:
#         raise ValueError("axis must be 0 or 1")
#
#     return out


def combine_detectors(detectors: list) -> np.array:
    """
    We combine detectors like in SNAP app:
    If pixels of 2 detectors overlap, we take the pixel of highest detector.
    """
    view_values = detectors[0]
    for i in range(1, len(detectors), 1):
        # view_values[np.isnan(view_values)] = detectors[i][np.isnan(view_values)]
        mask = np.isnan(view_values) & ~np.isnan(detectors[i])
        view_values[mask] = detectors[i][mask]

    return view_values


def fill_nans(arr):
    def nanmean_filter(values):
        center = values[len(values) // 2]  # center pixel
        if not np.isnan(center):
            return center  # keep original value if not NaN

        # neighbors except center
        neighbors = np.delete(values, len(values) // 2)
        valid = neighbors[~np.isnan(neighbors)]

        if len(valid) > 0:
            return valid.mean()
        else:
            return np.nan  # keep NaN if no valid neighbors


    # Apply with a 3x3 window
    return generic_filter(arr, nanmean_filter, size=3, mode='constant', cval=np.nan)


def interp_nans(arr):
    # grid of coordinates
    x, y = np.indices(arr.shape)

    # mask of valid points
    mask = ~np.isnan(arr)

    # interpolate onto missing points
    arr_interp = arr.copy()
    arr_interp[~mask] = griddata(
        (x[mask], y[mask]),  # known coordinates
        arr[mask],  # known values
        (x[~mask], y[~mask]),  # coords to interpolate
        method="nearest"  # "linear", "nearest", or "cubic"
    )
    return arr_interp

def upscale(arr, factor=500):
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


def get_mean_view_angles(root: ET.Element) -> tuple[np.array, np.array]:
    '''Extract view azimuth zenith values'''

    ns = {"n": "https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Tile_Metadata.xsd"}

    view_zenith_all = {}
    view_azimuth_all = {}
    for grid in root.findall(".//Band_Viewing_Incidence_Angles_Grids_List"):
        band = grid.attrib.get("band_id")

        detectors_a = []
        detectors_z = []
        for node in grid.findall("Viewing_Incidence_Angles_Grids"):
            view_azimuth_values = []
            view_zenith_values = []
            detector_id = node.attrib.get("detector_id")
            # print("detector", detector_id)
            for azimuth in node.findall('Azimuth/Values_List/VALUES'):
                row = [float(v) if v != 'NaN' else np.nan for v in azimuth.text.split()]
                view_azimuth_values.append(row)

            for zenith in node.findall('Zenith/Values_List/VALUES'):
                row = [float(v) if v != 'NaN' else np.nan for v in zenith.text.split()]
                view_zenith_values.append(row)
            detectors_a.append(np.array(view_azimuth_values))
            detectors_z.append(np.array(view_zenith_values))

        view_azimuth_values = (combine_detectors(detectors_a))
        view_zenith_values = (combine_detectors(detectors_z))

        view_azimuth_all[band] = view_azimuth_values
        view_zenith_all[band] = view_zenith_values

    mean_view_zenith = np.array(list(view_zenith_all.values()), dtype=np.float32).mean(0)
    mean_view_azimuth = np.array(list(view_azimuth_all.values()), dtype=np.float32).mean(0)

    # return interp_nans(interp_nans(mean_view_zenith)), interp_nans(interp_nans(mean_view_azimuth))
    return upscale((fill_nans_axiswise(mean_view_zenith[:22, :22]))), upscale((fill_nans_axiswise(mean_view_azimuth[:22, :22])))



def get_all_angles(root: ET.Element) -> Angles:
    """Get vie angles per tile per day"""

    sun_zenith_array = get_zun_zenith(root)
    sun_azimuth_array = get_zun_azimuth(root)

    mean_view_zenith, mean_view_azimuth = get_mean_view_angles(root)

    assert sun_zenith_array.shape == sun_azimuth_array.shape == mean_view_zenith.shape == mean_view_azimuth.shape

    return Angles(sun_zenith_array, sun_azimuth_array, mean_view_zenith, mean_view_azimuth)


def parse_geocoding(root: ET.Element, group_id: str = "R1") -> dict:
    """Pars georeference data from metadata file"""
    ns = {"n": root.tag.split("}")[0].strip("{")}  # detect namespace if present
    # --- Coordinate Reference System ---
    crs_elem = root.find(".//Geoposition_Informations/Coordinate_Reference_System/Horizontal_Coordinate_System", ns)
    epsg = crs_elem.find("HORIZONTAL_CS_CODE", ns).text
    # --- Group Geopositioning ---
    groups = root.findall(".//Group_Geopositioning", ns)
    geo = None
    for g in groups:
        if g.attrib.get("group_id") == group_id:
            geo = {
                "epsg": epsg,
                "ulx": float(g.find("ULX", ns).text),
                "uly": float(g.find("ULY", ns).text),
                "xdim": float(g.find("XDIM", ns).text),
                "ydim": float(g.find("YDIM", ns).text),
                "ncols": int(g.find("NCOLS", ns).text),
                "nrows": int(g.find("NROWS", ns).text),
            }
            break
    return geo


def make_angle_grid_coords(geo: dict, grid_shape) -> tuple[np.array, np.array]:
    """Get mesh grid with coordinates"""
    nrows, ncols = geo["nrows"], geo["ncols"]
    step_x = ncols / grid_shape[1]
    step_y = nrows / grid_shape[0]

    xs = [geo["ulx"] + (j * step_x + step_x / 2) * geo["xdim"] for j in range(grid_shape[1])]
    ys = [geo["uly"] + (i * step_y + step_y / 2) * geo["ydim"] for i in range(grid_shape[0])]

    xx, yy = np.meshgrid(xs, ys)

    utm_crs = CRS.from_epsg(geo["epsg"])  # example for tile in UTM zone 30N
    wgs84 = CRS.from_epsg(2154)
    transformer = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    xx, yy = transformer.transform(xx, yy)

    return xx, yy  # UTM coordinates


def build_angle_interpolator(angle_grid, geo, grid_shape):
    """
    angle_grid: numpy array
        - shape (23,23) for single angle
        - shape (23,23,N) for multiple angles (e.g. [zenith, azimuth])
    geo: dict with ULX, ULY, XDIM, YDIM, NCOLS, NROWS
    grid_shape: usually (23,23)
    """
    nrows, ncols = geo["nrows"], geo["ncols"]
    step_x = ncols / grid_shape[1]
    step_y = nrows / grid_shape[0]

    xs = [geo["ulx"] + (j * step_x + step_x / 2) * geo["xdim"] for j in range(grid_shape[1])]
    ys = [geo["uly"] + (i * step_y + step_y / 2) * geo["ydim"] for i in range(grid_shape[0])]

    utm_crs = CRS.from_epsg(geo["epsg"])  # example for tile in UTM zone 30N
    wgs84 = CRS.from_epsg(2154)
    transformer = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    xs, ys = transformer.transform(xs, ys)

    # RegularGridInterpolator expects strictly increasing coords
    if geo["ydim"] < 0:
        ys = ys[::-1]
        angle_grid = angle_grid[::-1, ...]

    interp = RegularGridInterpolator(
        (ys, xs),
        angle_grid,
        method="nearest",
        bounds_error=False,
        fill_value=None
    )
    return interp


def get_tile_info(tile: str) -> dict:
    base_path = "/your/path/input_data/patches"

    # nested defaultdict to simplify adding keys
    def nested_dict():
        return defaultdict(nested_dict)

    results_dict = nested_dict()

    for patch_id in os.listdir(os.path.join(base_path, tile)):
        patch_path = os.path.join(base_path, tile, patch_id)
        if not os.path.isdir(patch_path):
            continue

        img_list = [im for im in os.listdir(patch_path) if (im.endswith(".tif") and not im.startswith("MASK"))]
        for img_file in img_list:
            date = img_file.split('_')[1]
            results_dict[date][patch_id] = img_file

    return results_dict


def process_one_mtd(match_mtd: str, tile, id: dict) -> pd.DataFrame :
    path_mtd = os.path.join(path_metadata, tile, match_mtd)

    tree = ET.parse(path_mtd)
    root = tree.getroot()

    all_angles = get_all_angles(root)
    geo = parse_geocoding(root)

    coords_id = list(id.keys())
    file_names = list(id.values())

    coords = np.array([patch_id_to_coords(c) for c in coords_id])

    # Build interpolators
    interp = build_angle_interpolator(np.stack([*all_angles], axis=2), geo, all_angles.s_a.shape)
    center_values = interp(coords)

    angle_names = ["sun_zenith", "sun_azimuth", "view_zenith", "view_azimuth"]

    df = pd.DataFrame({
        "patch_id": coords_id,
        "image_name": file_names,
        "x": coords[:, 0],
        "y": coords[:, 1],
        **{name: np.round(center_values[:, i], 1) for i, name in enumerate(angle_names)}
    })

    return df

def patch_id_to_coords(patch_id: str) -> np.array:
    x_m, y_m = [int(v) * 1000 for v in patch_id.split('_')]
    return np.array([y_m - 500, x_m + 500])


def check_difference(df1, df2, angle, tol=3):
    diff = abs(df1[angle][~df2[angle].isnull()].values - df2[angle][~df2[angle].isnull()].values)
    diff = diff[~np.isnan(diff)]
    # print(diff[diff >= tol])
    # if not all(diff < tol):
    #     print(df1[angle][~df2[angle].isnull()].values)
    #     print(df2[angle][~df2[angle].isnull()].values)
    return all(diff < tol)

def compute_one_tile_angles(tile):
    output_path = "/your/path/input_data/sun_satellite_angles"
    metadata_files = os.listdir(os.path.join(path_metadata, tile))
    print(tile)
    tile_info = get_tile_info(tile)
    dfs = []
    for date, id in tile_info.items():
        # Parse
        matches_mtd = [f for f in metadata_files if f.__contains__(date) and not f.__contains__("_L3A_")]
        print("match", matches_mtd)

        if len(matches_mtd) == 1:
            match_mtd = matches_mtd[0]
            df = process_one_mtd(match_mtd, tile, id)
        elif len(matches_mtd) == 2:
            assert all("SENTINEL2A" in x for x in matches_mtd) or all("SENTINEL2B" in x for x in matches_mtd)
            for match_mtd in matches_mtd:
                df1 = process_one_mtd(matches_mtd[0], tile, id)
                count_nans1 = df1.view_zenith.isnull().sum()
                df2 = process_one_mtd(matches_mtd[1], tile, id)
                count_nans2 = df2.view_zenith.isnull().sum()
                print(match_mtd, count_nans1, count_nans2, len(df2))
                df = df1 if count_nans1 < count_nans2 else df2

                if df.view_azimuth.isnull().sum() == 0:
                    print()
                    # if count_nans1 < count_nans2:
                    #     assert check_difference(df1, df2, "view_azimuth")
                    #     assert check_difference(df1, df2, "view_zenith")
                    # if count_nans1 > count_nans2:
                    #     assert check_difference(df2, df1, "view_azimuth")
                    #     assert check_difference(df2, df1, "view_zenith")

                else:
                    # assert check_difference(df1, df2, "view_azimuth")
                    # assert check_difference(df1, df2, "view_zenith")
                    # assert check_difference(df2, df1, "view_azimuth")
                    # assert check_difference(df2, df1, "view_zenith")

                    # print(df[df.view_azimuth.isnull()])
                    # print(df2[df.view_azimuth.isnull()] if count_nans1 < count_nans2 \
                    #         else df1[df.view_azimuth.isnull()])
                    df[df.view_azimuth.isnull()] = \
                        df2[df.view_azimuth.isnull()] if count_nans1 < count_nans2 \
                            else df1[df.view_azimuth.isnull()]
        else:
            assert IndexError


        dfs.append(df)

    if dfs:
        dfsss = pd.concat(dfs, ignore_index=True)
        # print(dfsss[dfsss.view_zenith.isnull()])
        print("Null angles", dfsss.view_zenith.isnull().sum())

        dfsss.to_csv(os.path.join(output_path, tile + ".csv"))
    else:
        print(f"Tile {tile} is empty")


if __name__ == "__main__":
    TILES = ['32TMN', '31TDG', '31TGJ', '31TEK', '30TYP', '32TNN', '31UDR', '30TXQ', '30UXV', '31TDN', '31TEJ', '31UCR',
             '32TLS', '31TCL', '31TEN', '31TDK', '31TFM', '31TFN', '30TXR', '32TML', '31TDH', '32ULU', '31TFH', '31TEM',
             '31UDQ', '31UFQ', '30TXN', '32TLP', '32UMU', '32ULV', '30TYN', '30TXS', '31UCP', '32TLN', '31TGK', '31UFP',
             '31UCQ', '31UEQ', '30TWP', '30TXT', '31UGQ', '31UDS', '31TGM', '31UGP', '31TFK', '31TFL', '30TXP', '32TLT',
             '31TEG', '30TWN', '32TLR', '31TCK', '30UXU', '32TNM', '31TDL', '30TYR', '31TGL', '32UMV', '32TLQ', '32TMM',
             '32TNL', '30UYU', '31UDP', '30TYT', '31UCS', '31TCJ', '30UVV', '31TGN', '31TFJ', '30TYQ', '31TCH', '30UYV',
             '30UVU', '31TEH', '31UEP', '31TDM', '31TGH', '31TEL', '30TYS', '31TDJ']

    for tile in TILES:
        compute_one_tile_angles(tile)
