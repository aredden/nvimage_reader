from hashlib import md5
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union
from loguru import logger
import numpy as np
from nvidia.dali.types import DALIDataType
from .dali_dir_reader import BasePipeline, InterpTypes, ImageReturnShape, ResizeMode
from os import getenv


def has_torch():
    try:
        import torch

        return True
    except Exception as e:
        return False


def has_cupy():
    try:
        import cupy

        return True
    except Exception as e:
        return False


if has_torch():
    import torch
else:
    torch = None

if has_cupy():
    import cupy
else:
    cupy = None

try:
    import cupy as cp
except Exception as e:
    cp = None

cachedir = Path(getenv("NVREADER_CACHE_PATH", "~/.nvreader/tmp_images")).expanduser()
if not cachedir.is_dir():
    cachedir.mkdir(parents=True)


def handle_is_path(path: Path, exts: List[str], recurse: bool = True, debug=False):
    if not isinstance(path, Path):
        path = Path(path)
    if not path.name.endswith(".txt"):
        if path.is_dir():
            globstr = "**/*.*" if recurse else "*.*"
            paths = [i.absolute() for i in path.glob(globstr) if i.suffix in exts]
            if debug:
                logger.debug(f"Resolved {len(paths)} image paths in {path}")
            hashes = md5(b"".join([i.name.encode() for i in paths])).hexdigest()
            imout = (cachedir) / f"{hashes}_images.txt"
            imout_ims = [f"{i.as_posix()} {idx}" for idx, i in enumerate(paths)]
            imout.write_text("\n".join(imout_ims))
            return imout, paths, list(range(len(imout_ims)))
        else:
            raise Exception(f"Unsupported path {path} type {type(path)}")
    else:
        ims = [
            Path(i.strip().rsplit(" ", 1)[0])
            for i in path.read_text().splitlines()
            if i.strip()
        ]
        ims_indices = [
            int(i.strip().rsplit(" ", 1)[1])
            for i in path.read_text().splitlines()
            if i.strip()
        ]
        assert len(ims) == len(ims_indices), f"Images and indices are not the same!"
        return path, ims, ims_indices


def handle_is_list(paths: List[Path], exts: List[str], debug=False):
    """
    Resolves a list of paths to a list of images, saves to text file at cachedir
    """

    paths = [(Path(p) if not isinstance(p, Path) else p) for p in paths]
    paths = [i.absolute() for i in paths if i.suffix in exts]
    if debug:
        logger.debug(f"Resolved {len(paths)} from paths list")
    hashes = md5(b"".join([i.name.encode() for i in paths])).hexdigest()
    imout = (cachedir) / f"{hashes}_images.txt"
    imout_ims = [f"{i.as_posix()} {idx}" for idx, i in enumerate(paths)]
    imout.write_text("\n".join(imout_ims))
    return imout, paths, list(range(len(imout_ims)))


def dtype_to_tensor_type(dalitype: DALIDataType):
    """
    Maps DALI types to torch types
    """

    return {
        DALIDataType.BOOL: torch.bool,
        DALIDataType.FLOAT: torch.float32,
        DALIDataType.FLOAT16: torch.float16,
        DALIDataType.FLOAT64: torch.float64,
        DALIDataType.INT16: torch.int16,
        DALIDataType.INT32: torch.int32,
        DALIDataType.INT64: torch.int64,
        DALIDataType.INT8: torch.int8,
        DALIDataType.UINT16: torch.uint16,
        DALIDataType.UINT32: torch.uint32,
        DALIDataType.UINT64: torch.uint64,
        DALIDataType.UINT8: torch.uint8,
    }.get(dalitype, torch.uint8)


class DirReader(object):
    def __init__(
        self,
        image_paths_or_dir: Union[List[PathLike], Path, str],
        recursive: bool = True,
        image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".webp"],
        device: int = 0,
        batch_size: int = 1,
        threads: int = 1,
        to_cupy: bool = False,
        to_torch: bool = False,
        return_paths: bool = False,
        return_labels: bool = False,
        resize_to: Union[int, Tuple[int, int]] = None,
        crop_to: Union[int, Tuple[int, int]] = None,
        interpolation: InterpTypes = InterpTypes.LANCZOS,
        resize_mode: ResizeMode = ResizeMode.NOT_SMALLER,
        return_color: ImageReturnShape = ImageReturnShape.RGB,
        antialias=True,
    ) -> None:
        self.return_labels = return_labels
        if isinstance(image_paths_or_dir, (list, tuple, set)):
            image_paths_or_dir = list(image_paths_or_dir)
            self.resolved_txtlist, self.resolved_paths, self.indices = handle_is_list(
                image_paths_or_dir, image_extensions
            )
        else:
            self.resolved_txtlist, self.resolved_paths, self.indices = handle_is_path(
                Path(image_paths_or_dir)
                if isinstance(image_paths_or_dir, str)
                else image_paths_or_dir,
                image_extensions,
                recurse=recursive,
            )
        if device >= 0:
            self.device = "gpu"
        else:
            self.device = "cpu"
        self.batch_size = batch_size
        self.pipeline = BasePipeline(
            self.resolved_txtlist,
            batch_size=self.batch_size,
            num_threads=threads,
            device_id=0 if device < 1 else device,
            use_cuda=device >= 0,
            resize_to=resize_to,
            crop_to=crop_to,
            resize_mode=resize_mode,
            interpolation=interpolation,
            return_type=return_color,
            antialias=antialias,
        )
        self.device_index = device
        self.index = 0
        self.pipeline.build()
        self.return_paths = return_paths

        self.to_cupy = to_cupy and self.device == "gpu" and cupy is not None
        self.to_torch = to_torch or self.device == "gpu" and torch is not None

        if self.to_cupy:
            assert (
                self.device_index > -1
            ), f"Do not set to_torch to True if device_index is not set to a valid gpu index!"
        if self.to_torch:
            self.torch_device = torch.device(
                f"cuda:{self.device_index}" if self.device_index > -1 else "cpu"
            )
        if self.device == "gpu":
            assert (
                self.device_index > -1
            ), f"Do not set device to gpu if device_index is not set to a valid gpu index!"
        self.output_fn = self.determine_output_fn()

    def __len__(self):
        return len(self.resolved_paths)

    def __iter__(self):
        self.pipeline.reset()
        self.index = 0
        return self

    def to_torch_fn(self, ims):
        ims = (
            [
                torch.as_tensor(i, device=self.torch_device, dtype=torch.uint8)
                for i in ims
            ]
            if self.batch_size > 1
            else torch.as_tensor(ims[0], device=self.torch_device, dtype=torch.uint8)
        )
        return ims

    def to_cupy_fn(self, ims):
        return (
            [cp.asarray(i, dtype=cp.uint8) for i in ims]
            if self.batch_size > 1
            else cp.asarray(ims[0]).astype(cp.uint8)
        )

    def to_numpy_fn(self, ims):
        if hasattr(ims, "as_cpu"):
            ims = ims.as_cpu()
        return (
            [i.asnumpy().astype(np.uint8) for i in ims.as_cpu()]
            if self.batch_size > 1
            else ims[0].asnumpy().astype(np.uint8)
        )

    def determine_output_fn(self):
        if self.to_cupy:
            output_fn = self.to_cupy_fn
        elif self.to_torch:
            output_fn = self.to_torch_fn
        elif self.device != "gpu":
            output_fn = self.to_numpy_fn
        else:
            raise Exception(
                f"Could not convert gpu tensors to cupy or torch, please install cupy or torch!"
            )
        return output_fn

    def __next__(self):
        if self.index < len(self):
            r = self.pipeline.run()
            self.index += self.batch_size
            ims = r[0]
            paths = None
            labels = None
            if self.return_labels:
                indices = r[1]
                if hasattr(indices, "as_cpu"):
                    indices = indices.as_cpu()
                indices = indices.as_array().flatten()
                labels = indices.tolist()
            if self.return_paths:
                indices = r[1]
                if hasattr(indices, "as_cpu"):
                    indices = indices.as_cpu()
                indices = indices.as_array().flatten()
                paths = [
                    self.resolved_paths[i % len(self.resolved_paths)] for i in indices
                ]

            ims = self.output_fn(ims)
            return {"images": ims, "paths": paths, "labels": labels}
        else:
            raise StopIteration
