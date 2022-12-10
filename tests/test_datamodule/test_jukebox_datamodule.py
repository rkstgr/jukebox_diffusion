import os

import pytest

from src.datamodule.jukebox_datamodule import JukeboxDataModule

pytestmark = pytest.mark.skipif(
    "MAESTRO_DATASET_DIR" not in os.environ.keys(),
    reason="Environment variable 'MAESTRO_DATASET_DIR' missing."
)


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


def test_jukebox_datamodule_train(datamodule):
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        assert batch.shape == (8, 2048, 64)
        break


def test_jukebox_datamodule_val(datamodule):
    val_loader = datamodule.val_dataloader()
    for batch in val_loader:
        assert batch.shape == (8, 2048, 64)
        break


def test_jukebox_datamodule_test(datamodule):
    test_loader = datamodule.test_dataloader()
    for batch in test_loader:
        assert batch.shape == (8, 2048, 64)
        break
