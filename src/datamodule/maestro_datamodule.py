from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.dataset.maestro_dataset import MaestroDataset


class MaestroDataModule(pl.LightningDataModule):
    """
    DataModule for Maestro dataset.

    Args:
        root_dir (str): Path to the root directory of the maestro dataset (/path_to/maestro-v3.0.0).
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
            pin_memory: bool = False,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_length = sample_length
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = MaestroDataset(self.root_dir, split="train", sample_length=self.sample_length)
        self.val_dataset = MaestroDataset(self.root_dir, split="validation", sample_length=self.sample_length)
        self.test_dataset = MaestroDataset(self.root_dir, split="test", sample_length=self.sample_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)


if __name__ == "__main__":
    """
    Env:
        node3 with 16 cores

    Settings:
        batch_size: 32
        num_workers: 16

    Results:
        Time for setup: 25.130s
        32%|███▏      | 580/1797 [02:45<04:43, 4.2it/s]
    """

    import os
    from tqdm import tqdm
    import time

    start = time.time()
    dataloader = MaestroDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        batch_size=32,
        num_workers=16,
    )
    dataloader.setup()
    print(f"Time for setup: {time.time() - start:.3f}s")

    # take the time for one epoch
    start = time.time()
    train_loader = dataloader.train_dataloader()
    for batch in tqdm(train_loader):
        pass

    print(f"Time for first epoch: {time.time() - start:.3f}s")

    start = time.time()
    for batch in tqdm(train_loader):
        pass

    print(f"Time for second epoch: {time.time() - start:.3f}s")
