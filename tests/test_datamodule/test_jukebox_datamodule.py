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


@pytest.fixture
def datamodule():
    datamodule = JukeboxDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=1,
        batch_size=8,
        num_workers=4,
        sequence_len=2048,
        samples_per_file=4,
    )
    datamodule.setup()
    return datamodule


@depends_on_maestro_dataset
def test_jukebox_datamodule_train(datamodule):
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        assert batch.shape == (8, 2048, 64)
        break


@depends_on_maestro_dataset
def test_jukebox_datamodule_val(datamodule):
    val_loader = datamodule.val_dataloader()
    for batch in val_loader:
        assert batch.shape == (8, 2048, 64)
        break


@depends_on_maestro_dataset
def test_jukebox_datamodule_test(datamodule):
    test_loader = datamodule.test_dataloader()
    for batch in test_loader:
        assert batch.shape == (8, 2048, 64)
        break
