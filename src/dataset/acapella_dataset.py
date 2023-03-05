from pathlib import Path
from typing import Optional, Union, Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.files_dataset import FilesAudioDataset

JukeboxSample = Union[torch.Tensor, Dict[int, torch.Tensor]]

class TokenClass:
    values = {}

    @classmethod
    def get_id(cls, token: str):
        return cls.values[token]

    @classmethod
    def get_token(cls, id: int):
        return list(cls.values.keys())[list(cls.values.values()).index(id)]

    @classmethod
    def get_all_ids(cls):
        return list(cls.values.values())

class AcapellaLanguage(TokenClass):
    values = {
        "": 0,
        "Arabic": 1,
        "Assamese": 2,
        "Croatian": 3,
        "English": 4,
        "Greek": 5,
        "Hindi": 6,
        "Indonesian": 7,
        "Italian": 8,
        "Kannada": 9,
        "Malayalam": 10,
        "Persian": 11,
        "Portuguese": 12,
        "Spanish": 13,
        "Tamil": 14,
        "Turkish": 15,
        "Ukrainian": 16
    }

class AcapellaGender(TokenClass):
    values = {
        "": 0,
        "Female": 1,
        "Male": 2
    }

class AcapellaDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split: str,
            sample_length: Optional[int] = None,
            mono: bool = True,
            aug_shift: bool = False,
            initial_shuffle: bool = False
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
        self.metadata_file = self.root_dir / "dataset.csv"
        assert self.metadata_file.exists(), f"Metadata file {self.metadata_file} does not exist."
        self.metadata = pd.read_csv(self.metadata_file)

        # split
        generator = np.random.default_rng(42)
        self.metadata["split"] = generator.choice(["train", "validation", "test"], size=len(self.metadata), p=[0.8, 0.1, 0.1])
        self.metadata = self.metadata.query("split == @self.split")
        self.metadata = {
            r["ID"]: {
                "Singer": r["Singer"],
                "Language": r["Language"],
                "Gender": r["Gender"]
            }
            for _, r in self.metadata.iterrows()
        }

        # load singer_ids
        self.singer_ids = self.load_ids(self.root_dir / "singer_ids.csv")
        self.language_ids = self.load_ids(self.root_dir / "language_ids.csv")
        self.gender_ids = self.load_ids(self.root_dir / "gender_ids.csv")

        self.audio_dataset = FilesAudioDataset(root_dir=self.root_dir, sr=44100, channels=2, sample_length=sample_length,
                                         aug_shift=aug_shift, whitelist=self.metadata.keys())
        if initial_shuffle:
            # generate a random permutation of the indices
            self.indices = torch.randperm(len(self.audio_dataset))
        self.initial_shuffle = initial_shuffle

    def load_ids(self, path):
        df = pd.read_csv(path, header=None)
        # dict of name -> id
        return {row[0]: torch.tensor(row[1], dtype=torch.long) for row in df.itertuples(index=False)}

    def __len__(self):
        return len(self.audio_dataset)

    def __getitem__(self, index) -> JukeboxSample:
        if self.initial_shuffle:
            index = self.indices[index]

        audio, filename = self.audio_dataset[index]
        labels = self.metadata[Path(filename).stem]
        audio_sample = torch.from_numpy(audio)

        if self.mono:
            audio_sample = torch.mean(audio_sample, dim=1, keepdim=True)

        # audio_sample: (sample_length, channels)
        return {
            "audio": audio_sample,
            "singer_id": self.singer_ids.get(labels["Singer"], torch.tensor(0)),
            "language_id": self.language_ids.get(labels["Language"], torch.tensor(0)),
            "gender_id": self.gender_ids.get(labels["Gender"], torch.tensor(0))
        }


if __name__ == '__main__':
    import os
    import torchaudio

    dataset = AcapellaDataset(root_dir=os.environ["ACAPELLA_DATASET_DIR"], split="train", sample_length=44100*10, initial_shuffle=True)
    print(len(dataset))
    sample = dataset[0]
    print(sample["audio"].shape)
    print(sample["singer_id"])
    print(sample["language_id"])
    print(sample["gender_id"])
    # save sample
    torchaudio.save("sample.wav", sample["audio"].transpose(0, 1), 44100)

