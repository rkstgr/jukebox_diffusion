import os

from src.dataset.maestro_dataset import JukeboxDataset
from tests.conftest import depends_on_maestro_dataset

SEQUENCE_LEN = 2048
DATA_DIMENSION = 64

@depends_on_maestro_dataset
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

@depends_on_maestro_dataset
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

@depends_on_maestro_dataset
def test_maestro_dataset_multi_samples():
    dataset = JukeboxDataset(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=2,
        split="train",
        use_cache=False,
        sequence_len=SEQUENCE_LEN,
        samples_per_file=2,
    )

    sample = dataset[0]
    assert sample.shape == (2, SEQUENCE_LEN,     DATA_DIMENSION)

@depends_on_maestro_dataset
def test_maestro_dataset_multi_samples_multi_lvl():
    dataset = JukeboxDataset(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=[2, 1, 0],
        split="train",
        use_cache=False,
        sequence_len=SEQUENCE_LEN,
        samples_per_file=2,
    )

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert sample[2].shape == (2, SEQUENCE_LEN//16, DATA_DIMENSION)
    assert sample[1].shape == (2, SEQUENCE_LEN//4,  DATA_DIMENSION)
    assert sample[0].shape == (2, SEQUENCE_LEN,     DATA_DIMENSION)