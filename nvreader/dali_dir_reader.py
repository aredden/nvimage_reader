import types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
from nvidia.dali.types import DALIImageType
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from enum import Enum


class InterpTypes(Enum):
    CUBIC = types.INTERP_CUBIC
    GAUSSIAN = types.INTERP_GAUSSIAN
    LANCZOS = types.INTERP_LANCZOS3
    LINEAR = types.INTERP_LINEAR
    NEAREST = types.INTERP_NN


class ImageReturnShape(Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    YCbCr = "ycbcr"
    ANY_DATA = "any_data"

    def to_dali_type(self) -> DALIImageType:
        return {
            "rgb": DALIImageType.RGB,
            "bgr": DALIImageType.BGR,
            "gray": DALIImageType.GRAY,
            "ycbcr": DALIImageType.YCbCr,
            "any_data": DALIImageType.ANY_DATA,
        }.get(self.value)


class ResizeMode(Enum):
    NOT_SMALLER = "not_smaller"
    STRETCH = "stretch"
    NOT_LARGER = "not_larger"


def BasePipeline(
    file_list,
    batch_size=16,
    num_threads=2,
    device_id=0,
    seed=12,
    resize_to=None,
    crop_to=None,
    use_cuda=False,
    resize_mode: ResizeMode = ResizeMode.NOT_SMALLER,
    interpolation: InterpTypes = InterpTypes.LANCZOS,
    return_type: ImageReturnShape = ImageReturnShape.RGB,
    antialias=True,
):
    @pipeline_def(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed
    )
    def dataloader_graph():
        images, labels = fn.readers.file(
            name="Reader", file_list=file_list, device="cpu"
        )
        images = fn.decoders.image(
            images,
            device="mixed" if use_cuda else "cpu",
            output_type=return_type.to_dali_type(),
        )
        if resize_to is not None:
            images = fn.resize(
                images,
                size=resize_to,
                mode=resize_mode.value,
                interp_type=interpolation.value,
                antialias=antialias,
                device="gpu" if use_cuda else "cpu",
            )
        if crop_to is not None:
            images = fn.crop(
                images,
                crop=[crop_to, crop_to] if isinstance(crop_to, int) else crop_to,
            )

        return images, labels

    pipe: Pipeline = dataloader_graph()
    return pipe
