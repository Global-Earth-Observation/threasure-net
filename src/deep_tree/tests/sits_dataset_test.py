import random

import torch
from deep_tree.datamodules.dataset_sits import DeepTreeSITSMultiTileDataset, batch_data_collate_padding_fn
from deep_tree.datamodules.datatypes import DeepTreeSingleTileConfig


def get_dataset():
    return DeepTreeSITSMultiTileDataset(
        db_folder="/your/path/input_data/patches",
        lidar_folder="/your/path/input_data/lidarHD/lidarHD_processed",
        lidar_mask_folder="/your/path/input_data/lidarHD/lidarHD_mask_5m",
        angles_folder="/your/path/input_data/sun_satellite_angles",
        s2_folder="/your/path/input_data/patches",
        tiles=["30TXT", "31TDL"],
        lidarhd_metrics="None",  # TODO: useless for the moment
        context="training",
        config=DeepTreeSingleTileConfig(),
        max_patches_per_site=None,
        ref_date="year"
    )


def test_dataset():
    dataset = get_dataset()
    print(dataset)
    assert len(dataset) > 0
    assert dataset[0] is not None
    # ['input_tensor', 'target_tensor', 'input_tensor_mask', 'target_tensor_mask', 'full_mask', 'name_s2', 'name_lidar',
    #  'doy_s2', 'doy_lidar', 'pad_mask']
    for _ in range(10):
        idx = random.randrange(len(dataset))
        assert dataset[idx].doy_s2.tolist() == sorted(dataset[idx].doy_s2.tolist())
        assert len(dataset[idx].input_tensor) == len(dataset[idx].doy_s2)  == len(dataset[idx].name_s2)
        #== len(dataset[idx].input_tensor_mask)
        assert torch.all(dataset[idx].target_tensor[~dataset[idx].target_tensor_mask] >=0)


def test_dataloader():
    """Test dataloader and batch collator"""
    dl = torch.utils.data.DataLoader(
        get_dataset(),
        batch_size=2,
        drop_last=True,
        num_workers=1,
        collate_fn=batch_data_collate_padding_fn,
        shuffle=True,
        prefetch_factor=4,
        pin_memory=True,
    )
    batch = next(iter(dl))
    assert len(batch.input_tensor) == dl.batch_size

    # Assert correct padding
    assert len(batch.input_tensor[0]) == len(batch.doy_s2[0]) == len(batch.pad_mask[0])
    #==len(batch.input_tensor_mask[0])
    assert torch.all(batch.input_tensor[batch.pad_mask] == 0)
    assert torch.all(batch.doy_s2[batch.pad_mask] == 0)
