from pathlib import Path
from typing import List, Union, Dict
import re

import torch

"""
IDEA:

1. Load all files
"""

TARGET_SEQ_LEN = 4096


def load(root_dir) -> Dict[Path, List[Path]]:
    """
    1. List all files
    
    """
    root_dir = Path(root_dir)
    files = list(root_dir.glob("**/*lvl2.pt"))
    groups = {}
    for f in files:
        source_file = f.name.split(".")[0]
        if source_file not in groups:
            groups[source_file] = []
        groups[source_file].append(f)
    
    for source_file, parts in groups.items():
        groups[source_file] = sorted(parts)

    return groups


def resplit(files: List[Union[str, Path]]):
    data = torch.cat([torch.load(f, map_location="cpu") for f in files], dim=1)
    splits = torch.split(data, TARGET_SEQ_LEN, dim=1)
    file = files[0]
    total_parts = len(splits)
    for i, s in tqdm(enumerate(splits)):
        part = i+1
        new_file = file.parent / re.sub(r"part\d+-of-\d+", f"part{part:03d}-of-{total_parts:03d}", file.name)
        new_file = new_file.with_suffix(".v2.pt")
        torch.save(s.clone(), new_file)


if __name__ == "__main__":
    import os
    from tqdm import tqdm

    root_dir = os.environ["MAESTRO_DATASET_DIR"]

    print("Loading files...")
    groups = load(root_dir)
    print("Done")

    print("Resplitting...")
    progress = tqdm(groups.items())
    for source_file, parts in progress:
        resplit(parts)
