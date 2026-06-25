"""Inference code for different resolutions for pH95"""

# pylint: disable=C0103,W1203

import argparse
import logging
import os
import re
import xml.etree.ElementTree as ET
from argparse import Namespace
from typing import Tuple, Any

import hydra
import numpy as np
import pandas as pd
import rasterio
import torch
from omegaconf import OmegaConf
from rasterio.transform import Affine

from torchsisr import patches

from bin.compute_angles import build_angle_interpolator, get_all_angles, parse_geocoding
from deep_tree.datamodules.dataset_sits import transform_angles, batch_data_collate_padding_fn
from deep_tree.datamodules.datatypes import BatchData

torch.set_float32_matmul_precision("high")

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

PATH_MODELS = \
    "/your/path/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/"

def init_model(
        ckpt_path: str,
        cfg_path: str
) -> (torch.nn.Module, Tuple[torch.Tensor, torch.Tensor]):
    """Initialize SR model"""
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    checkpoint = torch.load(
        ckpt_path, weights_only=False, map_location=device
    )

    model_checkpoint = {
        k[6:]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model")
    }

    config = OmegaConf.load(cfg_path)

    # We instantiate the checkpoint configuration
    net = hydra.utils.instantiate(config.model.model)
    net.load_state_dict(model_checkpoint, strict=True)

    mean = torch.tensor(config.training_module.standardization_parameters.mean)
    std = torch.tensor(config.training_module.standardization_parameters.std)

    target_mean = torch.tensor(config.training_module.target_standardization_parameters.mean)
    target_std = torch.tensor(config.training_module.target_standardization_parameters.std)

    net.mean = mean
    net.std = std

    return net, (target_mean, target_std)


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for Catboost
    """
    arg_parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Inference",
    )

    arg_parser.add_argument(
        "--res",
        type=float,
        help="Output model resolution",
        default=10  # required=True
    )

    arg_parser.add_argument(
        "--patch_size",
        type=float,
        help="S2 patch size without margins",
        default=512  # required=True
    )

    arg_parser.add_argument(
        "--batch_size",
        type=float,
        help="Batch size for inference",
        default=10  # required=True
    )

    arg_parser.add_argument(
        "--tile",
        type=str,
        help="tile",
        default="31UDQ"
        # required=True
    )

    arg_parser.add_argument(
        "--year",
        type=int,
        help="year",
        default=2018
        # required=True
    )

    arg_parser.add_argument(
        "--max_len_sits",
        type=int,
        help="Maximum SITS length",
        default=12
        # required=True
    )

    arg_parser.add_argument(
        "--delta_day",
        type=int,
        help="Delta to 1 July",
        default=0
        # required=True
    )

    arg_parser.add_argument(
        "--tile_dir",
        type=str,
        help="Path to tiles",
        default="/your/path//THREASURE/inference/"
        # required=True
    )

    arg_parser.add_argument(
        "--out_dir",
        type=str,
        help="Output dir",
        default="/your/path//THREASURE/inference/"
        # required=True
    )

    return arg_parser


def save_array_as_tiff(
        array: np.ndarray,
        profile: dict[str, Any],
        output_path: str,
        resolution: float | int = 10.0,
        nodata: str | float = -10000
) -> None:
    """
    Save numpy array as GeoTIFF using the metadata of a reference raster.
    array shape: (bands, height, width)
    """
    rmin, rmax, cmin, cmax = compute_valid_bbox(array)
    print(rmin, rmax, cmin, cmax)
    array = array[:, rmin:rmax + 1, cmin:cmax + 1]

    xmin = profile["transform"].c
    ymax = profile["transform"].f

    # Compute new corner
    new_xmin = xmin + cmin * resolution
    new_ymax = ymax - rmin * resolution

    # Build new transform
    new_transform = Affine(resolution, 0, new_xmin,
                           0, -resolution, new_ymax)

    profile = {
        "driver": "GTiff",
        "dtype": array.dtype,
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "crs": profile["crs"],  # add CRS if known
        "transform": new_transform,
        "nodata": nodata
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array)

    print("Saved:", output_path)


def build_patch_grid(
        reference_raster_path: str, patch_size: int, margin: int, factor: int = 1,
        mask: np.ndarray | None = None
) -> tuple[list, dict[str, Any]]:
    """
    Returns:
        patches: list of dicts with:
            - row (top-left row)
            - col (top-left col)
            - corners: { 'tl', 'tr', 'bl', 'br' } with (x, y)
            - center: (x, y)
        meta: raster metadata
    """

    with rasterio.open(reference_raster_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        px = abs(transform.a)
        py = abs(transform.e)
        profile = src.profile

    if factor > 1:
        margin = int(margin / factor)

    patch_size_wm = patch_size + 2 * margin

    patches = []
    count = 0
    for r_ in np.arange(margin, height, patch_size):
        if r_ + patch_size + margin >= height:
            continue
        for c_ in np.arange(margin, width, patch_size):
            if c_ + patch_size + margin >= width:
                continue

            # We check if at least one pixel of mask falls in patch if mask exists
            if mask is not None:
                if not np.any(mask[r_ : r_+patch_size, c_ : c_ + patch_size]):
                    continue

            # ----- compute geographic coordinates of corners -----
            # row, col → (x, y)
            r = r_ - margin
            c = c_ - margin
            x_min, y_max = transform * (c, r)
            x_max, y_min = transform * (c + patch_size_wm, r + patch_size_wm)

            # ----- center -----
            cx, cy = transform * (c + patch_size_wm / 2, r + patch_size_wm / 2)

            patch_info = {
                "patch_id": count,
                "row": r,
                "col": c,
                "corners": [x_min, y_min, x_max, y_max],  # left, bottom, right, top,
                "center": (cx, cy)
            }

            patches.append(patch_info)
            count += 1

    meta = {
        "width": width, "height": height,
        "transform": transform, "crs": crs,
        "px": px, "py": py,
        "profile": profile
    }

    return patches, meta


def process_one_mtd(
        path_mtd: str, coords: list, date: str
) -> pd.DataFrame:
    """Process one metadata file"""
    tree = ET.parse(path_mtd)
    root = tree.getroot()

    all_angles = get_all_angles(root)
    geo = parse_geocoding(root)

    coords = np.array(coords)

    # Build interpolators
    interp = build_angle_interpolator(np.stack([*all_angles], axis=2), geo, all_angles.s_a.shape)
    center_values = interp(coords)

    angle_names = ["sun_zenith", "sun_azimuth", "view_zenith", "view_azimuth"]

    df = pd.DataFrame({
        "patch_id": np.arange(len(coords)),
        "date": date,
        "x": coords[:, 0],
        "y": coords[:, 1],
        **{name: np.round(center_values[:, i], 1) for i, name in enumerate(angle_names)}
    })

    return df


def read_patch_from_all_tifs(tif_paths: list[str], corners: list[float]) -> torch.Tensor:
    """Read time series from list of paths"""
    patches = []

    for path in tif_paths:
        with rasterio.open(path) as src:
            # print("nodata", src.nodata)
            patch = src.read(window=rasterio.windows.from_bounds(*corners, transform=src.transform),
                             boundless=True, fill_value=src.nodata)
            patches.append(torch.Tensor(patch))
    return torch.stack(patches, 0)  # list of np arrays


def extract_date(filename: str) -> str | None:
    """Extract date from image name"""
    match = re.search(r"_(\d{8})-", filename)
    if match:
        return match.group(1)
    return None


def get_paths(
        tile_dir: str, year: int | str
) -> tuple[list[str], list[str], list[str]]:
    """Get paths of inputs"""
    path_year = os.path.join(tile_dir, str(year))
    images_path = os.path.join(path_year, "images")
    mask_path = os.path.join(path_year, "mask")
    meta_path = os.path.join(path_year, "meta")

    images_path = \
        [os.path.join(images_path, i) for i in os.listdir(images_path) if i.endswith(".tif")]
    mask_path = \
        [os.path.join(mask_path, i) for i in os.listdir(mask_path) if i.endswith(".tif")]
    meta_path = \
        [os.path.join(meta_path, i) for i in os.listdir(meta_path) if i.endswith(".xml")]

    images_path = sorted(images_path, key=extract_date)
    mask_path = sorted(mask_path, key=extract_date)
    meta_path = sorted(meta_path, key=extract_date)
    return images_path, mask_path, meta_path


def compute_valid_bbox(
        arr: np.ndarray, nodata: str | float = -10000
) -> tuple[float, float, float, float]:
    """
    Returns (row_min, row_max, col_min, col_max)
    or None if the whole array is nodata.
    """
    # Mask of valid pixels across all bands
    # arr shape = (bands, H, W)
    valid_mask = (arr != nodata).any(axis=0)  # shape (H, W)

    rows = valid_mask.any(axis=1)
    cols = valid_mask.any(axis=0)

    # If no valid pixel at all
    if not rows.any() or not cols.any():
        row_min, row_max = 0, arr.shape[1] - 1
        col_min, col_max = 0, arr.shape[2] - 1

    else:
        row_inds = np.where(rows)[0]
        col_inds = np.where(cols)[0]

        row_min, row_max = row_inds[0], row_inds[-1]
        col_min, col_max = col_inds[0], col_inds[-1]

    return row_min, row_max, col_min, col_max


def prepare_patch(
        patch: dict,
        imgs_path: list[str],
        masks_path: list[str],
        ang_df: pd.DataFrame
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Prepare one patch to form a batch"""

    logging.info(f"Encoding {patch['patch_id']}...")
    if patch["center"][0] < 355000:
        return

    # Get valid images
    mask_sits_ = read_patch_from_all_tifs(masks_path, patch["corners"])

    cloud_pixels = torch.isin(
        mask_sits_.squeeze(1),
        torch.tensor([6, 14, -10000],
                     device=mask_sits_.device)
    )
    fraction = cloud_pixels.float().mean(dim=[-1, -2])
    cloud_mask = ~(fraction > 0.1)

    valid_ratio_mask = (
            (mask_sits_ == 0).squeeze(1).to(torch.float32).mean(dim=[-1, -2]) > 0.99)
    valid_images = (cloud_mask.int() + valid_ratio_mask.int()) == 2
    mask_sits = mask_sits_[valid_images]
    # valid_mask = valid_ratio_mask >= 0.9
    if len(mask_sits) == 0:
        logging.info("Empty patch")
        return
    logging.info(f"Valid images before {len(mask_sits_)}")
    logging.info(f"Valid images {len(mask_sits)}")

    images_sits = read_patch_from_all_tifs(
        np.array(imgs_path)[valid_images], patch["corners"]
    )

    patch_angles_df = ang_df[ang_df.patch_id == patch["patch_id"]].sort_values('date')

    patch_angles_df = patch_angles_df.loc[valid_images.numpy()]

    angles = transform_angles(
        patch_angles_df.sun_zenith.values,
        patch_angles_df.sun_azimuth.values,
        patch_angles_df.view_zenith.values,
        patch_angles_df.view_azimuth.values
    )
    dates = patch_angles_df['doy'].values

    if np.isnan(patch_angles_df.view_zenith.values).any():
        images_sits = images_sits[~np.isnan(patch_angles_df.view_zenith.values)]
        dates = dates[~np.isnan(patch_angles_df.view_zenith.values)]
        angles = angles[~np.isnan(patch_angles_df.view_zenith.values)]

    # Boolean mask of where the value appears
    mask = images_sits <= -5000

    # Count per batch element
    counts_per_batch = mask.view(mask.shape[0], -1).sum(dim=1)

    # Batch indices that contain at least one -10000
    nan_ratio = counts_per_batch / (images_sits.shape[-1] ** 2) > 0.01
    batch_indices = torch.nonzero(nan_ratio).squeeze(1)
    if len(batch_indices) > 0:
        images_sits = images_sits[~nan_ratio]
        dates = dates[~nan_ratio]
        angles = angles[~nan_ratio]
    images_sits[images_sits < -5000] = 0
    # print("Counts per batch:", counts_per_batch)
    # print("Batch indices with -10000:", batch_indices)

    return images_sits, dates, angles



def select_dates(doys: list, max_len_sits: int):

    bin_edges = np.linspace(0, 366, max_len_sits + 1)

    def chose_day(i):

        start = bin_edges[i]
        end = bin_edges[i + 1]

        idx_in_bin = np.where((doys >= start) & (doys < end))[0]
        if len(idx_in_bin) == 0:
            return


        bin_center = (start + end) / 2
        return idx_in_bin[
            np.argmin(np.abs(doys[idx_in_bin] - bin_center))
        ]

    return sorted(
        idx
        for idx in (chose_day(i) for i in range(max_len_sits))
        if idx is not None
    )


def prepare_batch(
        patches_in_batch,
        imgs_path: list[str],
        masks_path: list[str],
        ang_df: pd.DataFrame,
        d: list,
        factor: float,
        max_len_sits: int
):
    """Prepare one batch for encoding"""
    row_list = []
    col_list = []
    batch_list = []

    for patch in patches_in_batch:
        images_sits, dates, angles = prepare_patch(
            patch, imgs_path, masks_path, ang_df
        )
        row, col = patch["row"], patch["col"]

        logging.info(f"Valid images used {images_sits.shape[0]}")
        if images_sits.shape[0] > max_len_sits:
            # sorted_idx = np.unique(
            #     np.linspace(
            #         0, images_sits.shape[0] - 1, max_len_sits
            #     ).astype(np.int8)
            # )
            sorted_idx = select_dates(dates, max_len_sits)

            batch = BatchData(
                input_tensor=images_sits[sorted_idx],
                doy_s2=torch.Tensor(dates)[sorted_idx],
                doy_lidar=d,
                pad_mask=None,
                angles=angles[sorted_idx],
            )
        else:
            batch = BatchData(
                input_tensor=images_sits,
                doy_s2=torch.Tensor(dates),
                doy_lidar=d,
                pad_mask=None,
                angles=angles,
            )

        batch_list.append(batch)
        # Rows and cols are computed for S2 image at 10m resolution.
        # If the prediction resolution is different,
        # we multiply rows and cols by resolution factor,
        # but the margin is already scaled
        row = int(row * factor)
        col = int(col * factor)
        col_list.append(col)
        row_list.append(row)

    return batch_list, col_list, row_list


def get_margin_and_factor(model: torch.nn.Module) -> tuple[int, float | int]:
    """Gat multiplication factor and patch margin"""
    margin = model.get_prediction_margin()
    factor = 1
    if model.upsample_module is not None:
        factor = model.upsample_module.upsampling_factor
    margin = int(np.ceil(margin / (8 * factor)) * (8 * factor))
    return margin, factor


def get_checkpoint(resolution: float | int | str) -> tuple[str, str]:
    if resolution == 10:
        checkpoint = f"{PATH_MODELS}10m/epoch_025_best.ckpt"
        hydra_conf = f"{PATH_MODELS}10m/config.yaml"
    elif resolution == 5:
        checkpoint = f"{PATH_MODELS}sr_5m/epoch_038_best.ckpt"
        hydra_conf = f"{PATH_MODELS}sr_5m/config.yaml"
    elif resolution == 2.5:
        checkpoint = f"{PATH_MODELS}sr_2.5m/epoch_040_best.ckpt"
        hydra_conf = f"{PATH_MODELS}sr_2.5m/config.yaml"
    else:
        raise NotImplementedError
    return checkpoint, hydra_conf


def prepare_angles_df(meta_path: list[str], center_coords) -> pd.DataFrame:
    """Prepare dataframe with angles"""
    list_df = []
    for metafile in meta_path:
        list_df.append(process_one_mtd(metafile, center_coords, extract_date(metafile)))
    angles_df = pd.concat(list_df)

    date_format = pd.to_datetime(angles_df['date'], format='%Y%m%d')
    angles_df['doy'] = (date_format -
                        pd.to_datetime(date_format.dt.year.astype(str) + "-01-01")
                        ).dt.days
    return angles_df


def create_pred_image_names(args: argparse.Namespace) -> str:
    """Generate name for predicted image"""
    d = args.delta_day
    if d > 0:
        delta = "_d+" + str(d)
    elif d < 0:
        delta = "_d-" + str(abs(d))
    else:
        delta = "_d0"

    return f"{args.tile}_{args.year}{delta}_{args.res}m"


def encode_batch(
        batch_list: list,
        model: torch.nn.Module,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        margin: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one batch"""

    batch = batch_data_collate_padding_fn(batch_list)
    logging.info("Batch constructed for inference")

    model.eval()
    with torch.no_grad():
        pred_patch = model(batch)[:, :, margin:-margin, margin:-margin]
        classif_patch = pred_patch.classification
        pred_patch = patches.unstandardize(
            pred_patch.height,
            target_mean, target_std
        ).to(torch.int16).cpu()

    return pred_patch, classif_patch


def save_images(
        args: Namespace,
        output_path: str,
        meta: dict[str, Any],
        pred_tile: torch.Tensor,
        classif_tile: torch.Tensor | None = None,
        postfix: str | None = None
) -> None:
    """Save predicted images"""
    name_pred = create_pred_image_names(args)
    if postfix is not None:
        name_pred += postfix
    save_array_as_tiff(pred_tile.numpy(), meta["profile"],
                       os.path.join(
                           output_path,
                           f"Pred_{name_pred}.tif"
                       ),
                       float(args.res))

    if classif_tile is not None:
        save_array_as_tiff(classif_tile.numpy(), meta["profile"],
                           os.path.join(
                               output_path,
                               f"Pred_cl_{name_pred}.tif"
                           ),
                           float(args.res))

def write_batch(
        pred_patch: torch.Tensor,
        classif_patch: torch.Tensor,
        pred_tile: torch.Tensor,
        classif_tile: torch.Tensor,
        col_list: list[int],
        row_list: list[int],
        margin: int,
        patch_size_out: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write one batch results"""
    for pp in range(pred_patch.shape[0]):
        col, row = col_list[pp], row_list[pp]

        pred_tile[:, row + margin: row + margin + patch_size_out,
        col + margin: col + margin + patch_size_out] = pred_patch[pp]

        if classif_tile is not None:
            classif_tile[:, row + margin: row + margin + patch_size_out,
            col + margin: col + margin + patch_size_out] = classif_patch[pp]

    return pred_tile, classif_tile


def create_empty_tiles(
        meta: dict[str, Any],
        factor: int | float, classif: bool,
        nodata: float | int | None = -10000,
        bands_nb: int = 1
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Create empty tiles that will be filled with predictions"""
    pred_tile = torch.full(
        size=[bands_nb, int(meta['height'] * factor), int(meta['width'] * factor)],
        fill_value=nodata,
        dtype=torch.int32
    )

    if classif:
        classif_tile = torch.full(
            size=[1, int(meta['height'] * factor), int(meta['width'] * factor)],
            fill_value=nodata,
            dtype=torch.int32
        )
    else:
        classif_tile = None

    return pred_tile, classif_tile


if __name__ == "__main__":

    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    logging.info(f"Encoding tile {args.tile}")
    logging.info(f"Year {args.year}.")

    # Get model
    checkpoint, hydra_conf = get_checkpoint(args.res)
    model, (target_mean, target_std) = init_model(checkpoint, hydra_conf)

    # Define variables
    patch_size = args.patch_size    # input patch size
    margin, factor = get_margin_and_factor(model)
    batch_size = int(args.batch_size)
    patch_size_out = int(patch_size * factor)   # output patch size

    tile_dir = os.path.join(
        args.tile_dir,
        args.tile
    )
    output_path = os.path.join(
        args.out_dir,
        args.tile, str(args.year)
    )

    images_path, mask_path, meta_path = get_paths(tile_dir, str(args.year))

    ref_image_path = mask_path[0]

    patches_sits, meta = build_patch_grid(
        reference_raster_path=ref_image_path,
        patch_size=patch_size,
        margin=margin,
        factor=factor
    )
    center_coords = [patch["center"][::-1] for patch in patches_sits]

    angles_df = prepare_angles_df(meta_path, center_coords)

    masks = torch.stack(
        [
            read_patch_from_all_tifs([ref_image_path], p["corners"]).squeeze(0)
            for p in patches_sits
        ]
    )

    logging.info(f"Total patches {len(patches_sits)}.")

    # We eliminate patches that correspond to water bodies
    nb_pix = (patch_size + 2 * margin) ** 2
    not_to_use = (
            torch.isin(
                masks.squeeze(1),
                torch.Tensor([16, 30, 22])).sum(dim=(1, 2)) / nb_pix > 0.75
    ).numpy()
    patches_to_encode = np.array(patches_sits, dtype=object)[~not_to_use]

    pred_tile, classif_tile = create_empty_tiles(
        meta=meta,
        factor=factor,
        classif=model.regression.classif,
    )

    # We create batches from the whole image to encode
    for bb in range(int(np.ceil(len(patches_to_encode) / batch_size))):

        batch_list, col_list, row_list = prepare_batch(
            patches_in_batch=patches_to_encode[bb * batch_size: (bb + 1) * batch_size],
            imgs_path=images_path,
            masks_path=mask_path,
            ang_df=angles_df,
            d=args.delta_day + 183,
            factor=factor,
            max_len_sits=args.max_len_sits
        )

        if batch_list:
            pred_patch, classif_patch = encode_batch(
                batch_list, model, target_mean, target_std, margin
            )

            pred_tile, classif_tile = write_batch(
                pred_patch,
                classif_patch,
                pred_tile,
                classif_tile,
                col_list,
                row_list,
                margin,
                patch_size_out
            )

    save_images(args, output_path, meta, pred_tile, classif_tile)
