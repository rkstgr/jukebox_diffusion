import os

import pytest
import torch

from src.datamodule.maestro_datamodule import MaestroDataModule
from tests.conftest import depends_on_maestro_dataset

BATCH_SIZE = 2
SAMPLE_LENGTH = 44100
DATA_DIMENSION = 1
TARGET_SHAPE = (BATCH_SIZE, SAMPLE_LENGTH, DATA_DIMENSION)

@pytest.fixture
def datamodule():
    datamodule = MaestroDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        batch_size=BATCH_SIZE,
        num_workers=0,
        sample_length=SAMPLE_LENGTH,
    )
    datamodule.setup()
    return datamodule


@depends_on_maestro_dataset
def test_train_dataloader(datamodule):
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        assert batch.shape == TARGET_SHAPE
        break


@depends_on_maestro_dataset
def test_val_dataloader(datamodule):
    val_loader = datamodule.val_dataloader()
    for batch in val_loader:
        assert batch.shape == TARGET_SHAPE
        break


@depends_on_maestro_dataset
def test_test_dataloader(datamodule):
    test_loader = datamodule.test_dataloader()
    for batch in test_loader:
        assert batch.shape == TARGET_SHAPE
        break
