from frechet_audio_distance import FrechetAudioDistance
from pathlib import Path
import torch

frechet = FrechetAudioDistance(
    use_pca=False, 
    use_activation=False,
    verbose=False
)
if torch.cuda.is_available():
    print("Using CUDA")
    frechet.model = frechet.model.cuda()

fad_score = frechet.score(background_dir="eval/maestro/test", eval_dir="eval/maestro/validation_lvl0")

# **MAESTRO**
# Validation: 0.72
# Dance Diffusion: 11.54

# Validation Lvl0: 2.80
# Validation Lvl1: 7.35
# Unc Lvl1: Epoch  1: 56.1
# Unc Lvl1: Epoch 30: 19.8

print(f"FAD score: {fad_score}")
