from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.multiprocessing import Manager
from torch.utils.data import Dataset

JukeboxSample = torch.Tensor


class JukeboxDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split: str,
            lvl: int,
            sequence_len: Optional[int] = None,
            deterministic: bool = False,
            # only used when sequence_len is not None, will start the sample at the beginning of the embedding
            use_cache: bool = True,
            samples_per_file: int = 1,
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.deterministic = deterministic
        self.use_cache = use_cache
        self.samples_per_file = samples_per_file

        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), f"Root directory {self.root_dir} does not exist."
        self.lvl = lvl

        assert split in [
            "train",
            "validation",
            "test",
        ], f"Split must be one of 'train', 'validation', 'test', but got {split}"
        self.split = split

        # load the metadata
        self.metadata_file = self.root_dir / "maestro-v3.0.0.csv"
        assert self.metadata_file.exists(), f"Metadata file {self.metadata_file} does not exist."
        self.metadata = pd.read_csv(self.metadata_file).query("split == @self.split")

        self.file_paths = self.load_file_paths()

        if self.use_cache:
            self.cache = Manager().dict()

    def load_file_paths(self):
        # check locally if files array is cached
        cache_file = self.root_dir / f"files_cached_{self.split}_lvl{self.lvl}.csv"

        if cache_file.exists():
            file_paths = pd.read_csv(cache_file).values.flatten()
            return file_paths

        # all file_paths that have embeddings for this level and are in the split
        split_audio_files = set(self.metadata["audio_filename"].unique())
        file_paths = [f.relative_to(self.root_dir) for f in self.root_dir.glob(f"**/*lvl{self.lvl}*.pt")]
        file_paths = [f for f in file_paths if str(f).split(".")[0] + ".wav" in split_audio_files]

        # filter out the every file which correspond to the last sample
        # file has the form name.part{sample_nr}-of-{final_sample_nr}.jukebox.lvl0.pt
        # we want to keep all file_paths except the last one, where sample_nr == final_sample_nr
        def is_last_sample(file):
            part_desc = file.name.split(".")[1].split("-")
            start_index = part_desc[0][4:]
            end_index = part_desc[2]
            return start_index == end_index

        file_paths = [f for f in file_paths if not is_last_sample(f)]

        # cache the file_paths
        pd.DataFrame(file_paths).to_csv(cache_file, index=False)

        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file = self.file_paths[index]
        file = self.root_dir / file
        if self.use_cache and (file in self.cache):
            embedding = self.cache[file]
        else:
            embedding = torch.load(file, map_location=torch.device("cpu"))
            if self.use_cache:
                self.cache[file] = embedding
        if self.sequence_len is not None:
            if self.deterministic:
                embedding = torch.stack([embedding[..., : self.sequence_len, :] for _ in range(self.samples_per_file)],
                                        dim=0)
            else:
                # draw a random sample from the embedding
                embeddings = []
                for i in range(self.samples_per_file):
                    start_index = torch.randint(0, embedding.shape[-2] - self.sequence_len, (1,)).item()
                    embeddings.append(embedding[..., start_index: start_index + self.sequence_len, :])
                embedding = torch.stack(embeddings, dim=0)

        if embedding.ndim == 4:
            embedding = embedding.squeeze(1)

        if self.samples_per_file == 1:
            embedding = embedding.squeeze(0)

        return embedding.float()
