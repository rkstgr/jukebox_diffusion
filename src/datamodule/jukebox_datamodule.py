from typing import Optional, List
import warnings

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.dataset.jukebox_dataset import JukeboxSample, JukeboxDataset


def collate_fn(sample_list: List[JukeboxSample]):
    sample_list = [torch.unsqueeze(item, 0) if item.ndim == 2 else item for item in sample_list]
    return torch.cat(sample_list, dim=0)


class JukeboxDataModule(pl.LightningDataModule):
    """
    DataModule for Jukebox dataset.

    Args:
        root_dir (str): Path to the root directory of the dataset.
        lvl (int): Level of the dataset. 0 to 2.
            0: most detailed with ~5.5k samples per second.
            1: medium with ~1.4k samples per second.
            2: coarse with ~350 samples per second.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the data loader.
        sequence_len (int, optional): Length of the sequence to be sampled. Defaults to None, which means the entire sequence.
        samples_per_file (int, optional): Number of samples to be loaded from a single file. Defaults to 1.
            When setting this to >1 the dataset will return samples with (samples_per_file, sequence_len, 64). A custom collate_fn is used to
            concatenate the samples into a single tensor of shape (batch_size, sequence_len, 64).
            Note: This is useful for reducing the disc loading times as the entire file is loaded into memory. Not useful if the dataset is cached.
        pin_memory (bool, optional): Whether to pin the memory. Defaults to False.
        use_cache (bool, optional): Whether to cache the dataset. Defaults to True.
        *args: Arguments to be passed to the dataset.
        **kwargs: Keyword arguments to be passed to the dataset.
    
    """
    def __init__(
            self,
            root_dir,
            lvl: int,
            batch_size: int,
            num_workers: int,
            sequence_len: Optional[int] = None,
            samples_per_file: int = 1,
            pin_memory: bool = False,
            use_cache: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__()

        assert batch_size % samples_per_file == 0, f"Batch size {batch_size} must be divisible by samples per file {samples_per_file}"

        self.root_dir = root_dir
        self.lvl = lvl
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sequence_len = sequence_len
        self.samples_per_file = samples_per_file
        self.use_cache = use_cache
        self.args = args
        self.kwargs = kwargs

        if use_cache and num_workers == 0:
            warnings.warn("Cache is enabled but num_workers is 0. Please check if this was desired.")
        self.persistent_workers = True if use_cache and num_workers > 0 else False

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = JukeboxDataset(
            self.root_dir,
            "train",
            self.lvl,
            sequence_len=self.sequence_len,
            use_cache=self.use_cache,
            samples_per_file=self.samples_per_file,
            *self.args,
            **self.kwargs,
        )
        self.val_dataset = JukeboxDataset(
            self.root_dir,
            "validation",
            self.lvl,
            sequence_len=self.sequence_len,
            use_cache=self.use_cache,
            samples_per_file=self.samples_per_file,
            *self.args,
            **self.kwargs,
        )
        self.test_dataset = JukeboxDataset(
            self.root_dir,
            "test",
            self.lvl,
            sequence_len=self.sequence_len,
            use_cache=self.use_cache,
            samples_per_file=self.samples_per_file,
            *self.args,
            **self.kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.samples_per_file,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size // self.samples_per_file,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size // self.samples_per_file,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    """
    Env:
        node2 with 19 cores and 140GB RAM

    Settings:
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=2,
        batch_size=32,
        num_workers=8,
        sequence_len=2048*4,
        samples_per_file=2,
        use_cache=True,

    Results:
        Time for setup: 0.095s
        100%|██████████| 69/69 [00:42<00:00,  1.61it/s]
        Time for first epoch: 42.985s
        100%|██████████| 69/69 [00:02<00:00, 29.80it/s]
        Time for second epoch: 2.316s
    """
    
    import os
    from tqdm import tqdm
    import time

    start = time.time()
    dataloader = JukeboxDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=2,
        batch_size=32,
        num_workers=8,
        sequence_len=2048*4,
        samples_per_file=2,
        use_cache=True,
    )
    dataloader.setup()
    print(f"Time for setup: {time.time() - start:.3f}s")

    # take the time for one epoch
    start = time.time()
    train_loader = dataloader.val_dataloader()
    for batch in tqdm(train_loader):
        pass
    
    print(f"Time for first epoch: {time.time() - start:.3f}s")

    start = time.time()
    for batch in tqdm(train_loader):
        pass

    print(f"Time for second epoch: {time.time() - start:.3f}s")
