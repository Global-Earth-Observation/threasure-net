"""Pytorch Lightning DataSet block"""

import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from deep_tree.datamodules.datatypes import DeepTreeSingleTileConfig, BatchData
from deep_tree.datamodules.prepare_sample import RandomPatchCrop

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def compute_padding(batch_data: list[BatchData]) -> torch.Tensor:
    """Compute padding mask for time series"""
    lengths = torch.Tensor([element.input_tensor.shape[0] for element in batch_data])
    max_length = max(lengths)
    return torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)


def element_collator(
    batch_data: list[BatchData], el_name: str, padding: bool = True
) -> None | list | torch.Tensor:
    """
    Universal collator for each element of batch
    """
    collated_list = [getattr(element, el_name) for element in batch_data]

    if all(v is None for v in collated_list):
        return None

    if padding:
        return pad_sequence(collated_list, batch_first=True)

    return collated_list


def batch_data_collate_padding_fn(batch_data: list[BatchData]) -> BatchData:
    """Overloaded collate_fn for BatchData.

    Parameters
    ----------
    batch_data: List[BatchData]

    Return
    ------
    BatchData
    """

    collated_batch = BatchData(
        *[
            element_collator(
                batch_data,
                key,
                padding=not (("name" in key) or ("doy_lidar" in key)),
            )
            for key in batch_data[0].__dict__.keys()
        ]
    )
    collated_batch.pad_mask = compute_padding(batch_data)

    return collated_batch


class DeepTreeSITSMultiTileDataset(torch.utils.data.Dataset):
    """Handle all tiles S2 dataset."""

    def __init__(
        self,
        db_folder: str,
        lidar_folder: str,
        lidar_mask_folder: str,
        s2_folder: str,
        angles_folder: str | None,
        tiles: list[str],
        lidarhd_metrics: str,
        context: str,
        config: DeepTreeSingleTileConfig,
        max_patches_per_site: int | None,
        ref_date: str = "2014-03-03",
    ):
        """Init.

        Parameters
        ----------
        db_folder: str
        season: str
        lidarhd_metrics: str
        context: str
        tiles: List[str] | None
        config: DeepTreeSingleTileConfig
        max_patches_per_site: int | None
        lidarhd_exclude_values: List[float] | None = None
        """
        super().__init__()

        self.__tiles = tiles
        logging.info(f"Tiles: {self.__tiles}")
        logging.info(f"NB Tiles: {len(self.__tiles)}")
        single_tile_datasets: list[
            DeepTreeSITSSingleTileDataset | torch.utils.data.Subset
        ] = [
            DeepTreeSITSSingleTileDataset(
                db_folder=db_folder,
                lidar_folder=lidar_folder,
                lidar_mask_folder=lidar_mask_folder,
                s2_folder=s2_folder,
                angles_folder=angles_folder,
                tile_name=tile,
                lidarhd_metrics=lidarhd_metrics,
                context=context,
                config=config,
                ref_date=ref_date,
            )
            for tile in self.__tiles
        ]
        if max_patches_per_site is not None:
            single_tile_datasets = [
                torch.utils.data.Subset(
                    element,
                    random.sample(
                        list(range(len(element))),
                        k=min(len(element), max_patches_per_site),
                    ),
                )
                for element in single_tile_datasets
            ]
        self.dataset: torch.utils.data.ConcatDataset = torch.utils.data.ConcatDataset(
            single_tile_datasets
        )

    def __len__(self):
        """Size of dataset"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> BatchData:
        """Getter."""
        return self.dataset[idx]


def transform_angles(
        s_z: list, s_a: list, v_z: list, v_a: list
) -> torch.Tensor:
    """
    S2 angles we need to compute LAI:
    cos Z_s2, cos Z_sun, cos(A_sun-A_s2)

    S2 angles we initially have in the set:
    cos Z_sun, cos A_sun, sin A_sun, cos Z_s2, cos A_s2, sin A_s2
    """

    s_z = torch.deg2rad(torch.Tensor(s_z))
    s_a = torch.deg2rad(torch.Tensor(s_a))
    v_z = torch.deg2rad(torch.Tensor(v_z))
    v_a = torch.deg2rad(torch.Tensor(v_a))

    s_z_cos = torch.cos(s_z)
    s_a_cos = torch.cos(s_a)
    s_a_sin = torch.sin(s_a)

    v_z_cos = torch.cos(v_z)
    v_a_cos = torch.cos(v_a)
    v_a_sin = torch.sin(v_a)

    return torch.stack([s_z_cos, s_a_cos, s_a_sin, v_z_cos, v_a_cos, v_a_sin], 1)
    # return torch.round(angles[:, :, None, None].expand(-1, -1, *shape_s2[-2:]))


class DeepTreeSITSSingleTileDataset(torch.utils.data.Dataset):
    """Handle a single S2 tile dataset."""

    def __init__(
        self,
        db_folder: str,
        lidar_folder: str,
        lidar_mask_folder,
        s2_folder: str,
        angles_folder: str | None,
        tile_name: str,
        lidarhd_metrics: str,
        context: str,
        config: DeepTreeSingleTileConfig,
        ref_date: str = "2014-03-03",
    ):
        self.__lhd_exclude_values: list[float] | None = None
        if config.lidarhd_exclude_values is not None:
            # assert len(lidarhd_exclude_values) != 0
            self.__lhd_exclude_values = config.lidarhd_exclude_values

        self.ref_date = ref_date

        self.config = config

        self.s2_folder = s2_folder

        self.df = self.__prepare_df(
            db_folder, tile_name, lidar_mask_folder, angles_folder, context=context
        )
        self.__check_lidar(
            os.path.join(
                lidar_folder, tile_name, f"res_{self.config.target_resolution}"
            ),
            os.path.join(lidar_mask_folder, tile_name),
        )

        self.tile_name = tile_name

        self.sample_getter = RandomPatchCrop(
            lidar_folder=os.path.join(lidar_folder, tile_name),
            lidar_mask_folder=(
                None
                if lidar_mask_folder is None
                else os.path.join(lidar_mask_folder, tile_name)
            ),
            s2_folder=os.path.join(s2_folder, tile_name),
            cropped_s2_patch_size=self.config.cropped_s2_patch_size,  # without margin
            lidar_patch_size=self.config.lidar_patch_size,
            margin_s2=self.config.margin_s2,
            exclude_values_lhd=self.config.lidarhd_exclude_values,
            nodata_s2=self.config.nodata_s2,
            s2_resolution=self.config.input_resolution,
            lidar_resolution=self.config.target_resolution,
            training=str(context) == "training",
        )

        self.use_angles = angles_folder is not None

    def __len__(self) -> int:
        """Size of dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> BatchData | None:
        """Access to single item.

        Parameters
        ----------
        idx: int
        """
        item = self.df.iloc[idx]
        if self.config.max_len_sits is not None:
            item = self.__sample_row_by_max_len(item, self.sample_getter.training)
        prepared_item = self.sample_getter.get_sample(
            item.lidar_name, item.image_name, item.lidarhd_id, item.lidar_mask_name
        )
        prepared_item.doy_s2 = torch.Tensor(item.s2_doy)
        prepared_item.doy_lidar = item.lidar_doy

        if self.use_angles:
            prepared_item.angles = transform_angles(
                item.sun_zenith, item.sun_azimuth, item.view_zenith, item.view_azimuth
            )
        return prepared_item

    def __check_lidar(self, folder: str, lidar_mask_folder: str) -> None:
        """Check if path to files exist"""
        self.df = self.df[self.df["lidar_name"].isin(set(os.listdir(folder)))]
        # if not self.df.empty:
        #     self.df.loc[[not os.path.exists(os.path.join(lidar_mask_folder, name)) for name in
        #                  self.df["lidar_mask_name"]], "lidar_mask_name"] = None

    def __sample_row_by_max_len(self, row: pd.Series, training: bool) -> pd.Series:
        """Sample items with sits length > than certain value by masked pixels ratio"""
        length = len(row["s2_date"])
        if length <= self.config.max_len_sits:
            return row  # nothing to change

        if training:
            sorted_idx = sorted(random.sample(range(length), self.config.max_len_sits))
        else:
            sorted_idx = np.unique(
                np.linspace(0, length - 1, self.config.max_len_sits).astype(np.int8)
            )

        # subset each list
        cols_to_select = ["s2_date", "s2_doy", "image_name", "mask_name"]

        if self.use_angles:
            cols_to_select = (
                cols_to_select
                + ["sun_zenith"]
                + ["sun_azimuth"]
                + ["view_zenith"]
                + ["view_azimuth"]
            )

        for col in cols_to_select:
            row[col] = [row[col][i] for i in sorted_idx]

        return row

    def __compute_doy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform acquisition dates to timedelta with ref date"""
        df.loc[:, "s2_date"] = pd.to_datetime(df.s2_date)
        df.loc[:, "lidar_date"] = pd.to_datetime(df.lidar_date)

        if self.ref_date != "year":
            df.loc[:, "s2_doy"] = (df.s2_date - pd.to_datetime(self.ref_date)).dt.days
            df.loc[:, "lidar_doy"] = (
                df.lidar_date - pd.to_datetime(self.ref_date)
            ).dt.days
        else:
            df.loc[:, "s2_doy"] = (
                df.s2_date - df.s2_date.dt.to_period("Y").dt.start_time
            ).dt.days
            df.loc[:, "lidar_doy"] = (
                df.lidar_date - df.lidar_date.dt.to_period("Y").dt.start_time
            ).dt.days

        assert all((df.loc[:, "s2_doy"] - df.loc[:, "lidar_doy"]).abs() < 366)

        return df

    def __prepare_df(
        self,
        path: str,
        tile_name: str,
        lidar_mask_path: str | None,
        angles_path: str | None,
        context: str = "training",
    ):
        """Prepare dataset dataframe"""
        df = pd.read_parquet(os.path.join(path, tile_name, tile_name + ".parquet"))

        # Filter out non-valid images
        df = df[
            (df.masked_pixels <= self.config.max_masked_pixels)
            & (df.valid_pixels >= self.config.min_valid_pixels)
        ]

        # Choose context
        df = df[
            df["context"] == str(context)
        ]  # Need to cast context to string, otherwise does not work

        if df.empty:
            logging.info(
                f"Tile {tile_name} is empty from the beginning for {str(context)} task."
            )
            return df

        df = self.__compute_doy(df)

        if angles_path is not None:
            df_angles = pd.read_csv(
                os.path.join(angles_path, tile_name + ".csv"),
                usecols=[
                    "image_name",
                    "sun_zenith",
                    "sun_azimuth",
                    "view_zenith",
                    "view_azimuth",
                ],
            )

            df = pd.merge(
                df,
                df_angles,
                how="left",
                left_on="image_name",
                right_on="image_name",
            )
            df = df[~df.view_zenith.isnull()]

            # df = df[
            #     [os.path.exists(os.path.join(self.s2_folder, tile_name, name.split('.')[0][-9:], name))
            #      for name in df.image_name]]

            # print((df["image_name"][df.view_zenith.isnull()]))
            assert not df.view_zenith.isnull().values.any()
        # Group by 'ID' and sort Label by Value within each group
        # sort by ID and then Value

        df_sorted = df.sort_values(["lidarhd_id", "s2_date"])

        if angles_path is not None:
            grouped_lists = df_sorted.groupby("lidarhd_id").agg(
                {
                    "s2_date": list,
                    "s2_doy": list,
                    "image_name": list,
                    "mask_name": list,
                    "masked_pixels": list,
                    "sun_zenith": list,
                    "sun_azimuth": list,
                    "view_zenith": list,
                    "view_azimuth": list,
                }
            )
        else:
            grouped_lists = df_sorted.groupby("lidarhd_id").agg(
                {
                    "s2_date": list,
                    "s2_doy": list,
                    "image_name": list,
                    "mask_name": list,
                    "masked_pixels": list,
                }
            )

        other_columns = df_sorted.groupby("lidarhd_id").first()[
            ["lidar_name", "lidar_date", "lidar_doy", "tile_name"]
        ]

        aggregated_df = grouped_lists.join(other_columns).reset_index()
        if aggregated_df.empty:
            logging.info(f" Tile {tile_name} is empty for {str(context)} task.")
            return aggregated_df

        aggregated_df = aggregated_df[
            aggregated_df["s2_date"].apply(len) >= self.config.min_len_sits
        ]

        if aggregated_df.empty:
            logging.info(
                f"Tile {tile_name} is empty after SITS min length choosing for {str(context)} task."
            )
            return aggregated_df

        if lidar_mask_path is not None:
            df_mask = pd.read_parquet(
                os.path.join(
                    lidar_mask_path, tile_name, "mask_" + tile_name + ".parquet"
                )
            )
            aggregated_df = pd.merge(
                aggregated_df,
                df_mask,
                how="left",
                left_on="lidarhd_id",
                right_on="lidar_id",
            )
            len_before = len(aggregated_df)
            aggregated_df = aggregated_df[
                aggregated_df["valid_crop"] >= self.config.min_target_valid
            ]
            if aggregated_df.empty:
                logging.info(
                    f" Tile {tile_name} is empty after filtering {self.config.min_target_valid}% "
                    f"min target valid pixel ratio for {str(context)} task.."
                )
                return aggregated_df

            logging.info(
                f"Cleaned {int((len_before - len(aggregated_df)) / len_before * 100)}%"
                f" of targets with less "
                f"than {self.config.min_target_valid}% of valid pixels from tile {tile_name}."
            )
            aggregated_df = aggregated_df.drop(
                columns=["lidar_id", "tile", "valid_whole", "valid_crop"]
            )

        aggregated_df = aggregated_df.drop("masked_pixels", axis=1)

        # if self.config.max_len_sits is not None:
        #     aggregated_df = aggregated_df.apply(self.__sample_row_by_max_len, axis=1)

        return aggregated_df


class CacheDataset(torch.utils.data.Dataset):
    """
    A dataset that caches every retrieved tensor for later use
    """

    def __init__(self, dataset: torch.utils.data.Dataset):
        """ """
        super().__init__()
        self.dataset = dataset
        self.cache: dict[int, BatchData] = {}

    def __len__(self):
        """ """
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int):
        """ """
        if idx in self.cache:
            return self.cache[idx]

        ret = self.dataset[idx]
        self.cache[idx] = ret
        return ret
