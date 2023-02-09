import os

from src.dataset.maestro_dataset import MaestroDataset
from tests.conftest import depends_on_maestro_dataset

SEQUENCE_LEN = 44100

@depends_on_maestro_dataset
def test_maestro_dataset():
    dataset = MaestroDataset(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        split="train",
        sample_length=SEQUENCE_LEN
    )

    sample = dataset[0]
    assert sample.shape == (SEQUENCE_LEN, 1)