import types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

def BasePipeline(
    file_list,
    batch_size=16,
    num_threads=2,
    device_id=0,
    seed=12,
    resize_to=None,
    use_cuda=False
):
    @pipeline_def(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed
    )
    def dataloader_graph():
        images, labels = fn.readers.file(name='Reader', file_list=file_list, device="cpu")
        images = fn.decoders.image(images, device="mixed" if use_cuda else "cpu")
        if resize_to is not None:
            images = fn.resize(
                images,
                size=resize_to,
                mode="not_smaller",
                interp_type=types.INTERP_LANCZOS3,
                antialias=True,
                device="gpu" if use_cuda else "cpu",
            )
        return images, labels

    pipe: Pipeline = dataloader_graph()
    return pipe
