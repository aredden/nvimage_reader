from hashlib import md5
from os import PathLike
from pathlib import Path
from typing import List, Union
from loguru import logger
from .dali_dir_reader import BasePipeline
try:
    import cupy as cp
except Exception as e:
    cp = None

cachedir = Path("~/.nvreader/tmp_images").expanduser()
if not cachedir.is_dir():
    cachedir.mkdir(parents=True)


def handle_is_path(path: Path, exts: List[str], recurse: bool = True, debug=False):
    if not isinstance(path,Path):
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
        ims = [Path(i.rsplit(" ",1)[0]) for i in path.read_text().splitlines() if i.strip()]
        ims_indices = [int(i.rsplit(" ",1)[1]) for i in path.read_text().splitlines() if i.strip()]
        return path, ims, ims_indices

def handle_is_list(paths: List[Path], exts: List[str], debug=False):
    paths = [(Path(p) if not isinstance(p,Path) else p) for p in paths]
    paths = [i.absolute() for i in paths if i.suffix in exts]
    if debug:
        logger.debug(f"Resolved {len(paths)} from paths list")
    hashes = md5(b"".join([i.name.encode() for i in paths])).hexdigest()
    imout = (cachedir) / f"{hashes}_images.txt"
    imout_ims = [f"{i.as_posix()} {idx}" for idx, i in enumerate(paths)]
    imout.write_text("\n".join(imout_ims))
    return imout, paths, list(range(len(imout_ims)))

class DirectoryImageReader(object):
    def __init__(
        self,
        image_paths_or_dir: Union[List[PathLike], Path, str],
        recursive: bool = True,
        image_extensions=[".jpg", ".jpeg", ".png"],
        device=0,
        batch_size=1,
        threads=1,
        to_cupy=True,
    ) -> None:
        if isinstance(image_paths_or_dir,(list,tuple,set)):
            image_paths_or_dir = list(image_paths_or_dir)
            self.resolved_txtlist, self.resolved_paths, self.indices = handle_is_list(image_paths_or_dir, image_extensions)
        else:
            self.resolved_txtlist, self.resolved_paths, self.indices = handle_is_path(
                Path(image_paths_or_dir)
                if isinstance(image_paths_or_dir, str)
                else image_paths_or_dir,
                image_extensions,
                recurse=recursive,
            )
        if device >= 0:
            self.device='gpu'
        else:
            self.device='cpu'
        self.batch_size = batch_size
        self.pipeline = BasePipeline(
            self.resolved_txtlist,
            batch_size=self.batch_size,
            num_threads=threads,
            device_id=0 if device < 1 else device,
            use_cuda=device >= 0,
        )
        self.index = 0
        self.pipeline.build()
        self.to_cupy = to_cupy
    def __len__(self):
        return len(self.resolved_paths)
    
    def __iter__(self):
        self.pipeline.reset()
        self.index = 0
        return self
    
    def __next__(self):
        if self.index < len(self):
            r = self.pipeline.run()    
            self.index += self.batch_size
            ims = r[0]
            if self.device == 'gpu' and self.to_cupy:
                ims = [cp.asarray(i) for i in ims] if self.batch_size > 1 else cp.asarray(ims[0])
            else:
                ims = ims.as_cpu().as_array()
            return ims
        else:
            raise StopIteration