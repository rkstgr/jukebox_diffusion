# Create N samples of length L samples and evaluate the FAD score

import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
from scipy.io.wavfile import write

model_id = "harmonai/maestro-150k"
pipe = DiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")

TOTAL_SAMPLES = 64
BATCH_SIZE = 16  # max. 32 for 12GB GPU, saturates compute already at 8GB

samples = []
for i in range(TOTAL_SAMPLES // BATCH_SIZE):
    audios = pipe(batch_size=BATCH_SIZE, audio_length_in_s=5.0).audios
    samples.append(audios)

# To save locally
progress = tqdm(total=TOTAL_SAMPLES)
for i, sample in enumerate(samples):
    for j, audio in enumerate(sample):
        write(f"eval/maestro/dance_diffusion/test_{i*BATCH_SIZE+j}.wav", pipe.unet.sample_rate, audio.transpose())
        progress.update(1)
    