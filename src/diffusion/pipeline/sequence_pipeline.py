from diffusers import DiffusionPipeline


class SequencePipeline(DiffusionPipeline):
    def __call__(self, unet, scheduler, *args, **kwargs):
        raise NotImplementedError("SequencePipeline is not implemented yet.")


