from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.dataset.acapella_dataset import AcapellaDataset


class AcapellaDataModule(pl.LightningDataModule):
    """
    DataModule for Acapella dataset (https://ipcv.github.io/Acappella/acappella/).

    Args:
        root_dir (str): Path to the root directory.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the data loader.
        sample_length (int, optional): Length of the sequence to be sampled. Defaults to 10 seconds.
    """

    def __init__(
            self,
            root_dir: str,
            batch_size: int,
            num_workers: int,
            sample_length: Optional[int] = 44100 * 10,
            shuffle_train: bool = True,
            train_aug_shift: bool = True,
            pin_memory: bool = False,
    ):
        super().__init__()

        self.sample_rate = 44100

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_length = sample_length
        self.shuffle_train = shuffle_train
        self.train_aug_shift = train_aug_shift
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = AcapellaDataset(self.root_dir, split="train", sample_length=self.sample_length, aug_shift=self.train_aug_shift, initial_shuffle=True)
        self.val_dataset = AcapellaDataset(self.root_dir, split="validation", sample_length=self.sample_length, initial_shuffle=True)
        self.test_dataset = AcapellaDataset(self.root_dir, split="test", sample_length=self.sample_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
