
## NVREADER - A Python package for reading from a list of paths to images, or reading all images from a directory


### Installation

```bash
python -m pip install git+https://github.com/aredden/nvimage_reader.git
```

### Usage


Each iteration of the reader returns a dictionary with the following keys:

images: A list of images read from the current path in the form of either `torch.Tensor` or `numpy.ndarray` or `cupy.ndarray`.
paths: An (optional) list of paths to the images read- requires `return_paths=True` in the constructor.
labels: An (optional) list of either indices or labels for the images read optionally- requires `return_labels=True` in the constructor.


```python
from nvreader import DirReader

# Read all images from a directory
reader = DirReader('path/to/images', to_torch=True)
for image in reader:
    torch_image_list = image['images']
    ...
    # your code here

# Read all images from a list of paths

reader = DirReader(['path/to/image1.jpg', 'path/to/image2.webp'], to_torch=True, return_paths=True)

for image in reader:
    torch_image_list = image['images'] # is List[torch.Tensor]
    paths = image['paths'] # is List[str]
    ...
    # your code here
    ...
```
