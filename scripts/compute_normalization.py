import os
from einops import rearrange
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

from src.datamodule.maestro_datamodule import MaestroDataModule
from src.dataset.acapella_dataset import AcapellaDataset
from src.dataset.maestro_dataset import MaestroDataset
from src.model.jukebox_normalize import JukeboxNormalizer
from src.model.jukebox_vqvae import JukeboxVQVAEModel



datasets = {
    "maestro": lambda: ConcatDataset([
        MaestroDataset(os.environ["MAESTRO_DATASET_DIR"], split="train", sample_length=44100*5, aug_shift=False),
        MaestroDataset(os.environ["MAESTRO_DATASET_DIR"], split="validation", sample_length=44100*5, aug_shift=False),
        MaestroDataset(os.environ["MAESTRO_DATASET_DIR"], split="test", sample_length=44100*5, aug_shift=False),
    ]),

    "acapella": lambda: AcapellaDataset(os.environ["ACAPELLA_DATASET_DIR"], split="all", sample_length=44100*20, aug_shift=False),
}

def main(dataset_name: str, embedding_lvl: int, subset: int = 10):
    device = "cuda"

    dataset = datasets[dataset_name]()
    # take subset of dataset
    sub_idx = list(range(0, len(dataset), subset))
    dataset = Subset(dataset, sub_idx)

    dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=4, pin_memory=True)

    normalizer = JukeboxNormalizer()
    vqvae = JukeboxVQVAEModel().to(device)

    apply_fn = lambda x: vqvae.encode((x["audio"] if isinstance(x, dict) else x).to(device), lvl=embedding_lvl).detach().to("cpu")

    sample_accumulator = []
    for batch in tqdm(dataloader):
        sample = apply_fn(batch)
        sample = rearrange(sample, "b t c -> (b t) c")
        sample_accumulator.append(sample)
    
    sample_accumulator = torch.cat(sample_accumulator, dim=0)
    print(sample_accumulator.shape)
    mean = torch.mean(sample_accumulator, dim=0)
    std = torch.std(sample_accumulator, dim=0)

    print(mean)
    print(std)
    normalizer.save_stats(f"normalizations/{dataset_name}_lvl_{embedding_lvl}.pt", (mean, std))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="maestro")
    parser.add_argument("--embedding_lvl", type=int, default=2)
    parser.add_argument("--subset", type=int, default=10)
    args = parser.parse_args()

    main(args.dataset, args.embedding_lvl, subset=args.subset)