from pathlib import Path
from typing import Iterable, Optional, Union, List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

JukeboxSample = Union[torch.Tensor, Dict[int, torch.Tensor]]


class JukeboxDataset(Dataset):
    """
    Args:
        lvl: the level of the embedding. Must be one of [0, 1, 2]
    """

    def __init__(
            self,
            root_dir,
            split: str,
            lvl: Union[int, List[int]],
            sequence_len: Optional[int] = None,
            deterministic: bool = False,
            use_cache: bool = True,
            samples_per_file: int = 1,
            *args,
            **kwargs
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.deterministic = deterministic
        self.use_cache = use_cache
        self.samples_per_file = samples_per_file

        if self.deterministic:
            import warnings
            warnings.warn("Deterministic mode is enabled.")

        assert self.samples_per_file >= 1, f"Samples per file must be at least 1, but got {self.samples_per_file}"

        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), f"Root directory {self.root_dir} does not exist."

        if isinstance(lvl, int):
            lvl = [lvl]
        self.lvl: List[int] = sorted(lvl)
        self.max_lvl = max(lvl)
        self.min_lvl = min(lvl)

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

        self.file_paths = {lvl: self.load_file_paths(lvl) for lvl in self.lvl}

        if self.use_cache:
            print("Caching files...")
            self.cache = dict()

    def load_file_paths(self, lvl):
        # check locally if files array is cached
        cache_file = self.root_dir / f"files_cached_{self.split}_lvl{lvl}.v2.csv"

        if cache_file.exists():
            file_paths = pd.read_csv(cache_file).values.flatten()
            return file_paths

        # all file_paths that have embeddings for this level and are in the split
        split_audio_files = set(self.metadata["audio_filename"].unique())
        file_paths = [f.relative_to(self.root_dir) for f in self.root_dir.glob(f"**/*lvl{lvl}.v2.pt")]
        file_paths = [f for f in file_paths if str(f).split(".")[0] + ".wav" in split_audio_files]

        # filter out the every file which correspond to the last sample
        # file has the form name.part{sample_nr}-of-{final_sample_nr}.jukebox.lvl0.pt
        # we want to keep all file_paths except the last one, where sample_nr == final_sample_nr
        def is_last_sample(file):
            part_desc = file.name.split(".")[1].split("-")
            start_index = part_desc[0][4:]
            end_index = part_desc[2]
            return start_index == end_index

        file_paths = sorted([f for f in file_paths if not is_last_sample(f)])

        # cache the file_paths
        pd.DataFrame(file_paths).to_csv(cache_file, index=False)

        return file_paths

    def __len__(self):
        return len(self.file_paths[self.lvl[0]])

    def load_file(self, file) -> torch.Tensor:
        """Loads the embedding from the file.
        :param file: the file to load the embedding from
        :return torch.Tensor (S, 64): the embedding
        """
        if self.use_cache and (file in self.cache):
            return self.cache[file]

        embedding = torch.load(file, map_location=torch.device("cpu"))  # (1, S, 64)
        embedding = embedding.squeeze(0)  # (S, 64)

        if self.use_cache:
            self.cache[file] = embedding
        
        return embedding

    def getitem(self, lvl, index, start_offsets: Optional[torch.LongTensor] = None, sequence_len: Optional[int] = None):
        file = self.file_paths[lvl][index]
        file = self.root_dir / file
        embedding = self.load_file(file)

        if sequence_len is None:
            sequence_len = embedding.shape[-2]

        if start_offsets is None and not self.deterministic:
            if self.sequence_len == embedding.shape[-2]:
                start_offsets = torch.zeros(self.samples_per_file, dtype=torch.long)
            else:
                start_offsets = torch.randint(0, embedding.shape[-2] - self.sequence_len, (self.samples_per_file,)).long()

        elif self.deterministic:
            start_offsets = torch.zeros(self.samples_per_file, dtype=torch.long)

        embeddings = []
        for offset in start_offsets:
            embeddings.append(embedding[offset:offset + sequence_len, :].float())

        if len(embeddings) == 1:
            return embeddings[0], start_offsets
        
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings, start_offsets


    def __getitem__(self, index) -> JukeboxSample:
        if len(self.lvl) == 1:
            embeddings, _ = self.getitem(lvl=self.lvl[0], index=index, sequence_len=self.sequence_len)
            return embeddings

        sample = {}

        # get the sample for the lowest lvl
        embedding, start_offsets = self.getitem(lvl=self.lvl[0], index=index, sequence_len=self.sequence_len)
        sample[self.lvl[0]] = embedding

        # get the other samples
        for lvl in self.lvl[1:]:
            lvl_difference = lvl - self.min_lvl
            sequence_len = self.sequence_len // (4 ** lvl_difference)
            offsets = torch.div(start_offsets, 4 ** lvl_difference, rounding_mode="floor")
            sample[lvl] = self.getitem(lvl, index, offsets, sequence_len)[0]

        return sample
