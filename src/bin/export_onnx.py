from typing import Tuple

import hydra
import numpy as np
import onnxruntime
import torch
from omegaconf import OmegaConf
from torch import Tensor

from deep_tree.datamodules.dataset_sits import batch_data_collate_padding_fn
from deep_tree.datamodules.datatypes import BatchData
from torchsisr import patches


def init_model(
        ckpt_path: str,
        cfg_path: str
) -> (torch.nn.Module, Tuple[torch.Tensor, torch.Tensor]):
    """Initialize SR model"""
    checkpoint = torch.load(
        ckpt_path, weights_only=False
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

def round_decimals(x, decimals=3):
    factor = 10 ** decimals
    return torch.round(x * factor) / factor

class ThreasureOnnx(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std
        self.model.eval()

    def unstd(self, height: Tensor) -> Tensor:
        height_unstd = patches.unstandardize(
            height, self.mean.to(height.device), self.std.to(height.device)
        ).clamp(min=0)

        return round_decimals(height_unstd, decimals=3)

    def forward(
        self, sits_s2: Tensor, doy_s2: Tensor, doy_lidar: Tensor, pad_mask: Tensor, angles: Tensor
    ):
        input_threasure = BatchData(
            input_tensor=sits_s2, doy_s2=doy_s2, doy_lidar=doy_lidar, pad_mask=pad_mask, angles=angles
        )
        out = self.model(input_threasure)
        return self.unstd(out.height), out.classification


# def get_ckpt(res: float|str):
#     if res == 10:
#         # checkpoint = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS/10m/epoch_023_val_total_loss=0.25961381.ckpt"
#         # hydra_conf = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS/10m/config.yaml"
#         checkpoint = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/10m/epoch_025_best.ckpt"
#         hydra_conf = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/10m/config.yaml"
#     elif res == 5:
#         checkpoint = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/sr_5m/epoch_038_best.ckpt"
#         hydra_conf = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/sr_5m/config.yaml"
#     elif res == 2.5:
#         checkpoint = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/sr_2.5m/epoch_040_best.ckpt"
#         hydra_conf = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/sr_2.5m/config.yaml"
#     return checkpoint, hydra_conf

def get_ckpt(res: float|str):
    if res == 10:
        # checkpoint = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS/10m/epoch_023_val_total_loss=0.25961381.ckpt"
        # hydra_conf = "/your/path/dev/deep-tree/training_experiments/BEST_MODELS/10m/config.yaml"
        checkpoint = "/home/ekalinicheva/sufosat/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/10m/epoch_025_best.ckpt"
        hydra_conf = "/home/ekalinicheva/sufosat/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/10m/config.yaml"
    elif res == 5:
        checkpoint = "/home/ekalinicheva/sufosat/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/sr_5m/epoch_038_best.ckpt"
        hydra_conf = "/home/ekalinicheva/sufosat/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/sr_5m/config.yaml"
    elif res == 2.5:
        checkpoint = "/home/ekalinicheva/sufosat/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/sr_2.5m/epoch_040_best.ckpt"
        hydra_conf = "/home/ekalinicheva/sufosat/dev/deep-tree/training_experiments/BEST_MODELS_CLASSIF/sr_2.5m/config.yaml"
    return checkpoint, hydra_conf

def get_margin(model):
    margin = model.get_prediction_margin()
    factor = 1
    if model.upsample_module is not None:
        factor = model.upsample_module.upsampling_factor
    margin = int(np.ceil(margin / (8 * factor)) * (8 * factor))
    return margin



    margin_final_pred = np.ceil(margin / (8 * factor)) * (8 * factor)
    assert margin.is_integer()


def main():
    n = 11
    h, w = 64, 64
    resolutions = [10, 5, 2.5]
    for r in resolutions:
        checkpoint, hydra_conf = get_ckpt(res=r)
        model, (mean, std) = init_model(checkpoint, hydra_conf)

        if torch.cuda.is_available():
            model = model.cuda()

        margin = get_margin(model)

        dummy_input1 = BatchData(
                    input_tensor=torch.randn(n, 10, h, w),
                    doy_s2=torch.randn(n),
                    doy_lidar=0,
                    pad_mask=None,
                    angles=torch.randn(n, 6),
                )
        dummy_input2= BatchData(
                    input_tensor=torch.randn(n+2, 10, h, w),
                    doy_s2=torch.randn(n+2),
                    doy_lidar=0,
                    pad_mask=None,
                    angles=torch.randn(n+2, 6),
                )

        dummy_batch = batch_data_collate_padding_fn([dummy_input1, dummy_input2, dummy_input1])
        if torch.cuda.is_available():
            dummy_batch = dummy_batch.to(device="cuda:0")
        dummy_input = (
            dummy_batch.input_tensor,
            dummy_batch.doy_s2,
            torch.tensor(dummy_batch.doy_lidar),
            dummy_batch.pad_mask,
            dummy_batch.angles
        )

        input_names = ["sits_s2", "doy_s2", "doy_lidar", "pad_mask", "angles"]
        output_names = ["height", "classification"]

        dynamic_axes = {
            "sits_s2": {0: "b", 1: "n", 3: "h", 4: "w"},
            "doy_s2": {0: "b", 1: "n"},
            "doy_lidar": {0: "b"},
            "pad_mask": {0: "b", 1: "n"},
            "angles": {0: "b", 1: "n"},
            "height": {0: "b", 1: "h", 2: "w"},
            "classification": {0: "b", 1: "h", 2: "w"},
        }


        threasure_model = ThreasureOnnx(model, mean, std)

        torch.onnx.export(
            threasure_model,
            dummy_input,
            f"./threasure_{r}m_margin_lidar_{margin}.onnx",
            opset_version=18,
            export_params=True,
            verbose=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        sess_opt = onnxruntime.SessionOptions()
        sess_opt.intra_op_num_threads = 16
        sess_opt.inter_op_num_threads = 16
        ort_session = onnxruntime.InferenceSession(
            f"./threasure_{r}m_margin_lidar_{margin}.onnx",
            sess_opt,
            providers=["CUDAExecutionProvider"],
        )

        dummy_batch = batch_data_collate_padding_fn([dummy_input1, dummy_input2, dummy_input1, dummy_input2])

        dummy_input = {
            "sits_s2": dummy_batch.input_tensor.cpu().numpy(),
            "doy_s2": dummy_batch.doy_s2.cpu().numpy(),
            "doy_lidar": np.array(dummy_batch.doy_lidar),
            "pad_mask": dummy_batch.pad_mask.cpu().numpy(),
            "angles": dummy_batch.angles.cpu().numpy()
        }

        # if torch.cuda.is_available():
        #     dummy_batch = dummy_batch.to(device="cuda:0")
        dummy_input_tensor = (
            dummy_batch.input_tensor,
            dummy_batch.doy_s2,
            torch.tensor(dummy_batch.doy_lidar),
            dummy_batch.pad_mask,
            dummy_batch.angles
        )

        # Check the ouput of the exported model
        height, classif = ort_session.run(["height", "classification"], dummy_input)
        with torch.no_grad():
            threasure_model.eval().to("cpu")
            torch_height, torch_classif = threasure_model(*dummy_input_tensor)


        print(
            torch.allclose(
                torch.tensor(height), torch_height.cpu(), atol=1e-02
            )
        )
        print(
            torch.allclose(
                torch.tensor(classif), torch_classif.cpu(), atol=1e-02
            )
        )

if __name__ == "__main__":
    main()
