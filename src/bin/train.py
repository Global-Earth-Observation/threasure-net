# -*- coding: utf-8 -*-
"""Training entrypoint."""

import logging
from pathlib import Path
from typing import Dict

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from torchsisr import hydra_utils


def extract_checkpoint(checkpoints: Dict, name: str) -> Dict:
    """Fetch checkpoint from name.

    Parameters
    ----------
    checkpoints: Dict
    name: str

    Return
    ------
    Dict
    """
    # checkpoint: Dict = {}
    # for element in checkpoints["state_dict"].keys():
    #     checkpoint.update(
    #         {
    #             element.split(".", maxsplit=2)[2]: v
    #             for element, v in checkpoints["state_dict"].items()
    #             if element.startswith(name)
    #         }
    #     )
    checkpoint = {
        element.split(".", maxsplit=2)[2]: v
        for element, v in checkpoints["state_dict"].items()
        if element.startswith(name)
    }
    return checkpoint


# pylint: disable=too-few-public-methods
class ConfigOverloader:
    """Overload config."""

    def __init__(self, config: DictConfig):
        """Init.

        Parameters
        ----------
        config: DictConfig
        """
        self.__config = config
        self.__set_model_name()
        self.__set_config_name()

    def __set_model_name(self) -> None:
        """Overload model name."""
        model_name = (
                self.__config.model.name
                + "_"
                + "x".join(
            str(element) for element in self.__config.model.regression.hidden_dims
        )
        )
        self.__config.model.name = model_name

    def __set_config_name(self) -> None:
        """Overload global __config name."""
        self.__config.name = (
                self.__config.model.name
                + "_"
                + self.__config.datamodule.name
                + "_"
                + self.__config.losses.name
        )

    def get_name(self) -> str:
        """Getter.

        Return
        ------
        str
        """
        return self.__config.name


# pylint: disable=too-many-branches
@hydra.main(version_base=None, config_path="../../hydra", config_name="main.yaml")
def main(config: DictConfig) -> None:
    """Main training processor.

    Parameters
    ----------
    config: DictConfig
    """
    hydra_utils.extras(config)
    # cfg_ovl = ConfigOverloader(config)
    # checkpoints_dir = os.path.join(config.work_dir, "checkpoints", cfg_ovl.get_name())
    # Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    # os.chdir(checkpoints_dir)
    logging.info("Current working directory: %s", Path.cwd())
    if config.get("loglevel"):
        numeric_level = getattr(logging, config.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {config.loglevel}")
        logging.basicConfig(
            level=numeric_level,
            datefmt="%y-%m-%d %H:%M:%S",
            format="%(asctime)s :: %(levelname)s :: %(message)s",
        )
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    if config.get("seed"):
        pl.seed_everything(config.get("seed"), workers=True)
    if config.get("mat_mul_precision"):
        torch.set_float32_matmul_precision(config.get("mat_mul_precision"))
    datamodule = hydra.utils.instantiate(config.datamodule.data_module)
    training_module = hydra.utils.instantiate(config.training_module.training_module)

    # Define callbacks
    # (from https://github.com/ashleve/lightning-hydra-template/blob/
    # a4b5299c26468e98cd264d3b57932adac491618a/src/training_pipeline.py#L50)
    callbacks: list[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                # pylint: disable=protected-access
                logging.info("Instantiating callback <%s>", cb_conf._target_)
                callbacks.append(hydra.utils.instantiate(cb_conf))
    loggers: list[pl.loggers.logger.Logger] = []
    if "loggers" in config:
        for _, lg_conf in config.loggers.items():
            if "_target_" in lg_conf:
                # pylint: disable=protected-access
                logging.info("Instantiating logger <%s>", lg_conf._target_)
                loggers.append(hydra.utils.instantiate(lg_conf))

    nb_training_batches = len(datamodule.train_dataloader())
    nb_validation_batches = len(datamodule.val_dataloader())

    logging.info(
        "nb_training_batches=%s, nb_validation_batches=%s",
        str(nb_training_batches),
        str(nb_validation_batches),
    )

    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers
    )

    # Train the model
    if config.get("train"):
        logging.info("Starting training!")
        if config.resume_from_checkpoint is not None:
            logging.info("Training from checkpoint %s", config.resume_from_checkpoint)
        else:
            logging.info("Training from scratch")

        trainer.fit(
            training_module, datamodule, ckpt_path=config.resume_from_checkpoint
        )

    if config.get("test"):
        test_ckpt_path: str | None = "best"

        if config.resume_from_checkpoint is not None and not config.get("train"):
            test_ckpt_path = config.resume_from_checkpoint

        logging.info("Starting testing with model %s", test_ckpt_path)

        if config.get("train"):
            # model already exists inside trainer
            trainer.test(
                datamodule=datamodule,
                ckpt_path=test_ckpt_path,
            )
        else:
            # test-only run → must pass model class
            trainer.test(
                model=training_module,
                datamodule=datamodule,
                ckpt_path=test_ckpt_path,
            )

    logging.info("Finalizing!")

    # Print path to best checkpoint
    if (
            config.resume_from_checkpoint is not None
            and trainer.checkpoint_callback is not None
            and hasattr(trainer.checkpoint_callback, "best_model_path")
    ):
        logging.info(
            "Best model ckpt at %s",
            trainer.checkpoint_callback.best_model_path,
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
