from typing import List
from argparse import Namespace, ArgumentParser
from datetime import datetime
import logging

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

#from mimo.utils import dir_path
#from mimo.models.mimo_unet import MimoUnetModel
#from mimo.tasks.sen12tp.sen12tp_datamodule import get_datamodule, add_datamodule_args
#from mimo.tasks.sen12tp.callbacks import OutputMonitor
from sen12tp_datamodule import get_datamodule, add_datamodule_args

from typing import List
from pathlib import Path

def dir_path(string) -> Path:
    """Helper for argument parsing. Ensures that the provided string is a directory."""
    path = Path(string)
    if path.is_dir():
        return path
    else:
        raise NotADirectoryError(string)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main(args: Namespace):
    
    dm = get_datamodule(args)
    print('dm', dm)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=dir_path,
        default="/deepskieslab/rnevin/zenodo_data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Specify the batch size.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Specify the patch size.",
    )
    args = parser.parse_args()  # Parse command-line arguments
    main(args)  # Call main function with parsed arguments