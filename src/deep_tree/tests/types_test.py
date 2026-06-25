"""Test suites for custom types."""

import pytest
import torch

from deep_tree.datamodules.datatypes import BatchData


def test_batch_data() -> None:
    """Test init BatchData."""
    input_tensor = torch.ones(size=(10, 10, 51, 51))
    target_tensor = torch.ones(size=(10, 1, 512, 512))
    _ = BatchData(
        input_tensor=input_tensor, target_tensor=target_tensor, target_tensor_mask=None
    )
    target_tensor_mask = torch.ones(size=(10, 1, 512, 512))
    _ = BatchData(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        target_tensor_mask=target_tensor_mask,
    )
    input_tensor = torch.ones(size=(10, 51, 51))
    target_tensor = torch.ones(size=(1, 512, 512))
    _ = BatchData(
        input_tensor=input_tensor, target_tensor=target_tensor, target_tensor_mask=None
    )
    target_tensor_mask = torch.ones(size=(1, 512, 512))
    _ = BatchData(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        target_tensor_mask=target_tensor_mask,
    )


def test_batch_data_device_cpu() -> None:
    """Test move batch to device."""
    input_tensor = torch.ones(size=(10, 10, 51, 51))
    target_tensor = torch.ones(size=(10, 1, 512, 512))
    batch = BatchData(
        input_tensor=input_tensor, target_tensor=target_tensor, target_tensor_mask=None
    )
    batch = batch.to(torch.device("cpu"))  # type: ignore
    assert batch.input_tensor.get_device() == -1
    assert batch.target_tensor.get_device() == -1
    target_tensor_mask = torch.ones(size=(10, 1, 512, 512))
    batch = BatchData(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        target_tensor_mask=target_tensor_mask,
    )
    batch = batch.to(torch.device("cpu"))  # type: ignore
    assert batch.target_tensor_mask.get_device() == -1  # type: ignore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_batch_data_device_cuda() -> None:
    """Test moove batch to device."""
    input_tensor = torch.ones(size=(10, 10, 51, 51))
    target_tensor = torch.ones(size=(10, 1, 512, 512))
    batch = BatchData(
        input_tensor=input_tensor, target_tensor=target_tensor, target_tensor_mask=None
    )
    batch = batch.to(torch.device("cuda"))  # type: ignore
    assert batch.input_tensor.is_cuda
    assert batch.target_tensor.is_cuda
    target_tensor_mask = torch.ones(size=(10, 1, 512, 512))
    batch = BatchData(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        target_tensor_mask=target_tensor_mask,
    )
    batch = batch.to(torch.device("cuda"))  # type: ignore
    assert batch.target_tensor_mask.is_cuda  # type: ignore
