import os

import pytest

from src.dataset.jukebox_dataset import JukeboxDataset

SEQUENCE_LEN = 2048
DATA_DIMENSION = 64

@pytest.mark.skipif(
    "MAESTRO_DATASET_DIR" not in os.environ.keys(),
    reason="Environment variable 'MAESTRO_DATASET_DIR' missing."
)
def test_maestro_dataset():
    dataset = JukeboxDataset(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=2,
        split="train",
        use_cache=False,
        sequence_len=SEQUENCE_LEN,
    )

    sample = dataset[0]
    assert sample.shape == (SEQUENCE_LEN, DATA_DIMENSION)

@pytest.mark.skipif(
    "MAESTRO_DATASET_DIR" not in os.environ.keys(),
    reason="Environment variable 'MAESTRO_DATASET_DIR' missing."
)
def test_maestro_dataset_multi_lvl():
    dataset = JukeboxDataset(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=[2, 1, 0],
        split="train",
        use_cache=False,
        sequence_len=SEQUENCE_LEN,
    )

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert sample[2].shape == (SEQUENCE_LEN//16, DATA_DIMENSION)
    assert sample[1].shape == (SEQUENCE_LEN//4,  DATA_DIMENSION)
    assert sample[0].shape == (SEQUENCE_LEN,     DATA_DIMENSION)
