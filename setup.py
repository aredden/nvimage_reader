from setuptools import setup



setup(
    name="nvreader",
    include_dirs=['nvreader'],
    requires=[
        "numpy",
        "pillow",
        "cupy_cuda12x",
        "opencv_python_headless",
        "nvidia_dali_cuda120"
    ]
)