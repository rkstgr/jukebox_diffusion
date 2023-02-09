import os

from torch.utils.data import DataLoader, ConcatDataset

from src.datamodule.maestro_datamodule import MaestroDataModule
from src.dataset.maestro_dataset import MaestroDataset
from src.model.jukebox_normalize import JukeboxNormalizer
from src.model.jukebox_vqvae import JukeboxVQVAEModel



datasets = {
    "maestro_val": MaestroDataset(os.environ["MAESTRO_DATASET_DIR"], split="validation", sample_length=44100*10, aug_shift=False),
    "maestro_train": MaestroDataset(os.environ["MAESTRO_DATASET_DIR"], split="train", sample_length=44100*10, aug_shift=False),
    "maestro_test": MaestroDataset(os.environ["MAESTRO_DATASET_DIR"], split="test", sample_length=44100*10, aug_shift=False),
    
    "maestro_all": ConcatDataset([
        MaestroDataset(os.environ["MAESTRO_DATASET_DIR"], split="train", sample_length=44100*10, aug_shift=False),
        MaestroDataset(os.environ["MAESTRO_DATASET_DIR"], split="validation", sample_length=44100*10, aug_shift=False),
        MaestroDataset(os.environ["MAESTRO_DATASET_DIR"], split="test", sample_length=44100*10, aug_shift=False),
    ]),
}

def main(dataset_name: str, embedding_lvl: int):
    device = "cuda"

    dataset = datasets[dataset_name]

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    normalizer = JukeboxNormalizer()
    vqvae = JukeboxVQVAEModel().to(device)

    apply_fn = lambda x: vqvae.encode(x.to(device), lvl=embedding_lvl).detach().to("cpu")

    mean, std = normalizer.compute_stats_iter(dataloader, apply_fn=apply_fn, total=len(dataloader))
    normalizer.save_stats(f"config/normalizations/{dataset_name}_lvl_{embedding_lvl}.pt", (mean, std))


if __name__ == "__main__":
    main("maestro_train", 1)