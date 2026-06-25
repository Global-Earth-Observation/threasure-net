"""Dataset module."""

import logging

import pytorch_lightning as pl
import torch

from .dataset_sits import (
    DeepTreeSITSMultiTileDataset,
    CacheDataset,
    batch_data_collate_padding_fn,
)
from .datatypes import DeepTreeDataModuleConfig


# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


class DeepTreeSITSDataModule(pl.LightningDataModule):
    """Datamodule."""

    def __init__(self, config: DeepTreeDataModuleConfig):
        """Init.

        Parameters
        ----------
        config: DeepTreeDataModuleConfig
        """

        super().__init__()

        self.config = config

        self.training_dataset: torch.utils.data.Dataset
        self.validation_dataset: torch.utils.data.Dataset | None = None
        self.testing_dataset: torch.utils.data.Dataset | None = None

        self.ref_date = self.config.ref_date

        # if self.config.testing_tiles is not None:
        #     self.testing_dataset = DeepTreeMultiTileDataset(
        #         db_folder=self.config.db_folder,
        #         season=self.config.season,
        #         lidarhd_metrics=self.config.lidarhd_metrics,
        #         context="test",
        #         tiles=self.config.testing_tiles,
        #         config=self.config.single_tile_dataset_config,
        #         max_patches_per_site=self.config.max_patches_per_site,
        #     )

        self.training_dataset = DeepTreeSITSMultiTileDataset(
            db_folder=self.config.db_folder,
            lidar_folder=self.config.lidar_folder,
            lidar_mask_folder=self.config.lidar_mask_folder,
            s2_folder=self.config.s2_folder,
            angles_folder=self.config.angles_folder,
            lidarhd_metrics=self.config.lidarhd_metrics,
            context="training",
            tiles=self.config.tiles,
            config=self.config.single_tile_dataset_config,
            max_patches_per_site=self.config.max_patches_per_site,
            ref_date=self.ref_date,
        )
        logging.info("Training dataset is done")

        if not self.config.stats_computation:
            self.validation_dataset = DeepTreeSITSMultiTileDataset(
                db_folder=self.config.db_folder,
                lidar_folder=self.config.lidar_folder,
                lidar_mask_folder=self.config.lidar_mask_folder,
                s2_folder=self.config.s2_folder,
                angles_folder=self.config.angles_folder,
                lidarhd_metrics=self.config.lidarhd_metrics,
                context="validation",
                tiles=self.config.tiles,
                config=self.config.single_tile_dataset_config,
                max_patches_per_site=self.config.max_patches_per_site,
                ref_date=self.ref_date,
            )
            logging.info("Validation dataset is done")

            self.testing_dataset = DeepTreeSITSMultiTileDataset(
                db_folder=self.config.db_folder,
                lidar_folder=self.config.lidar_folder,
                lidar_mask_folder=self.config.lidar_mask_folder,
                s2_folder=self.config.s2_folder,
                angles_folder=self.config.angles_folder,
                lidarhd_metrics=self.config.lidarhd_metrics,
                context="test",
                tiles=self.config.tiles,
                config=self.config.single_tile_dataset_config,
                max_patches_per_site=self.config.max_patches_per_site,
                ref_date=self.ref_date,
            )
            logging.info("Testing dataset is done")

            # Use a cache for validation, to seep up validation steps
            if config.cache_validation_dataset:
                self.validation_dataset = CacheDataset(self.validation_dataset)

            if config.cache_testing_dataset:
                self.testing_dataset = CacheDataset(self.testing_dataset)

        logging.info(
            "%i training patches available", len(self.training_dataset)
        )  # type: ignore

        if not self.config.stats_computation:
            logging.info(
                "%i validation patches available",
                len(self.validation_dataset),  # type: ignore
            )

            logging.info(
                "%i testing patches available", len(self.testing_dataset)  # type: ignore
            )

    def train_dataloader(self):
        """
        Return train dataloaded (reset every time this method is called)
        """
        return torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.config.batch_size,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=batch_data_collate_padding_fn,
            shuffle=True,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Return validation data loader (never reset)
        """
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.config.testing_validation_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=batch_data_collate_padding_fn,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Return test data loader (never reset)
        """
        return torch.utils.data.DataLoader(
            self.testing_dataset,
            batch_size=self.config.testing_validation_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=batch_data_collate_padding_fn,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )
