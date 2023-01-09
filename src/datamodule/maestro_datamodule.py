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
        
        
    On mac with 0 workers:
        4s/it
    """

    import os
    from tqdm import tqdm
    import time

    start = time.time()
    dataloader = MaestroDataModule(
        root_dir=os.environ["MAESTRO_DATASET_DIR"],
        batch_size=32,
        num_workers=0,
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
