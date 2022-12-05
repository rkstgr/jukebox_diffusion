import os

import pytest

from src.datamodule.jukebox_datamodule import JukeboxDataModule


@pytest.fixture
def datamodule():
    datamodule = JukeboxDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=1,
        batch_size=8,
        num_workers=4,
        num_samples=2048,
        samples_per_file=4,
    )
    datamodule.setup()


def test_jukebox_datamodule_train(datamodule):
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        assert batch.shape == (4, 2048, 64)
        break


def test_jukebox_datamodule_val(datamodule):
    val_loader = datamodule.val_dataloader()
    for batch in val_loader:
        assert batch.shape == (4, 2048, 64)
        break


def test_jukebox_datamodule_test(datamodule):
    test_loader = datamodule.test_dataloader()
    for batch in test_loader:
        assert batch.shape == (4, 2048, 64)
        break
