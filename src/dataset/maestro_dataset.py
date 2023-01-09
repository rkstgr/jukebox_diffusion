from pathlib import Path
from typing import Optional, Union, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataset.files_dataset import FilesAudioDataset

JukeboxSample = Union[torch.Tensor, Dict[int, torch.Tensor]]


class MaestroDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split: str,
            sample_length: Optional[int] = None,
            mono: bool = True,
    ):
        super().__init__()
        self.sample_length = sample_length

        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), f"Root directory {self.root_dir} does not exist."

        assert split in [
            "train",
            "validation",
            "test",
        ], f"Split must be one of 'train', 'validation', 'test', but got {split}"
        self.split = split
        self.mono = mono

        # load the metadata
        self.metadata_file = self.root_dir / "maestro-v3.0.0.csv"
        assert self.metadata_file.exists(), f"Metadata file {self.metadata_file} does not exist."
        self.metadata = pd.read_csv(self.metadata_file).query("split == @self.split")
        whitelist = [Path(f).stem for f in self.metadata.audio_filename]
        self.dataset = FilesAudioDataset(root_dir=self.root_dir, sr=44100, channels=2, sample_length=sample_length,
                                         min_duration_sec=10,
                                         filenames_whitelist=whitelist)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> JukeboxSample:
        audio_sample = torch.from_numpy(self.dataset[index])

        if self.mono:
            audio_sample = torch.mean(audio_sample, dim=1, keepdim=True)

        return audio_sample  # (sample_length, channels)


if __name__ == '__main__':
    import os
    import torch.distributed as dist

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    dataset = MaestroDataset(root_dir=os.environ["MAESTRO_DATASET_DIR"], split="train", sample_length=44100)
    print(len(dataset))
    print(dataset[0].shape)
