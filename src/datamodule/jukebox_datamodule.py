from typing import Optional, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.dataset.jukebox_dataset import JukeboxSample, JukeboxDataset


def collate_fn(sample_list: List[JukeboxSample]):
    sample_list = [torch.unsqueeze(item, 0) if item.ndim == 2 else item for item in sample_list]
    return torch.cat(sample_list, dim=0)


class JukeboxDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root_dir,
            lvl: int,
            batch_size: int,
            num_workers: int,
            num_samples: Optional[int] = None,
            samples_per_file: int = 1,
            persistent_workers: bool = False,
            pin_memory: bool = False,
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
        self.num_samples = num_samples
        self.persistent_workers = persistent_workers
        self.samples_per_file = samples_per_file
        self.args = args
        self.kwargs = kwargs

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = JukeboxDataset(
            self.root_dir,
            "train",
            self.lvl,
            num_samples=self.num_samples,
            use_cache=True,
            samples_per_file=self.samples_per_file,
            *self.args,
            **self.kwargs,
        )
        self.val_dataset = JukeboxDataset(
            self.root_dir,
            "validation",
            self.lvl,
            num_samples=self.num_samples,
            use_cache=True,
            samples_per_file=self.samples_per_file,
            *self.args,
            **self.kwargs,
        )
        self.test_dataset = JukeboxDataset(
            self.root_dir,
            "test",
            self.lvl,
            num_samples=self.num_samples,
            use_cache=True,
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
            persistent_workers=True,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size // self.samples_per_file,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size // self.samples_per_file,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    import os

    dataloader = JukeboxDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        lvl=1,
        batch_size=4,
        num_workers=4,
        num_samples=2048,
        samples_per_file=8,
    )

    dataloader.setup()
    train_loader = dataloader.train_dataloader()
    for batch in train_loader:
        print(batch.shape)
        break
