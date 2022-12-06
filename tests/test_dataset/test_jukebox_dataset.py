import os

import pytest

from src.dataset.jukebox_dataset import JukeboxDataset


@pytest.mark.skipif(
    "MAESTRO_DATASET_DIR" not in os.environ.keys(),
    reason="Environment variable 'MAESTRO_DATASET_DIR' missing."
)
def test_jukebox_dataset():
    dataset = JukeboxDataset(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=2,
        split="train",
        use_cache=False,
        sequence_len=2048,
        samples_per_file=1,
        deterministic=False
    )

    sample = dataset[0]
    assert sample.shape == (2048, 64)
