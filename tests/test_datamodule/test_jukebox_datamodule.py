import os

import pytest
import torch

from src.datamodule.jukebox_datamodule import JukeboxDataModule, collate_fn

depends_on_maestro_dataset = pytest.mark.skipif(
    "MAESTRO_DATASET_DIR" not in os.environ.keys(),
    reason="Environment variable 'MAESTRO_DATASET_DIR' missing."
)


def test_collate_fn():
    x = [torch.rand(2048, 64) for _ in range(8)]
    assert collate_fn(x).shape == (8, 2048, 64)


def test_collate_fn_multi_lvl():
    lvl = [2, 1, 0]
    x = [{lvl: torch.rand(2048, 64) for lvl in lvl} for _ in range(8)]
    batch = collate_fn(x)
    assert isinstance(batch, dict)
    assert set(batch.keys()) == set(lvl)
    assert batch[2].shape == (8, 2048, 64)

BATCH_SIZE = 2
SEQUENCE_LEN = 2048
DATA_DIMENSION = 64
TARGET_SHAPE = (BATCH_SIZE, SEQUENCE_LEN, DATA_DIMENSION)

@pytest.fixture
def datamodule():
    datamodule = JukeboxDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=2,
        batch_size=BATCH_SIZE,
        num_workers=0,
        sequence_len=SEQUENCE_LEN,
        samples_per_file=1,
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

@depends_on_maestro_dataset
def test_multiple_lvl():
    datamodule = JukeboxDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=[2, 1, 0],
        batch_size=BATCH_SIZE,
        num_workers=0,
        sequence_len=SEQUENCE_LEN,
        samples_per_file=1,
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        assert isinstance(batch, dict)
        assert set(batch.keys()) == set([2, 1, 0])
        assert batch[2].shape == (BATCH_SIZE, SEQUENCE_LEN//16, DATA_DIMENSION)
        assert batch[1].shape == (BATCH_SIZE, SEQUENCE_LEN//4,  DATA_DIMENSION)
        assert batch[0].shape == (BATCH_SIZE, SEQUENCE_LEN,     DATA_DIMENSION)
        break

@depends_on_maestro_dataset
@pytest.mark.skip(reason="Not implemented yet")
def test_multiple_samples_per_file():
    datamodule = JukeboxDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=2,
        batch_size=BATCH_SIZE,
        num_workers=0,
        sequence_len=SEQUENCE_LEN,
        samples_per_file=2,
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        assert batch.shape == TARGET_SHAPE
        break