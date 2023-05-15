from setuptools import setup, find_packages
print(find_packages("."))


setup(
    name="nvreader",
    author="Alex Redden",
    version="0.0.1",
    py_modules=["nvreader.read_dir", "nvreader.dali_dir_reader",'nvreader.__init__'],
    include_dirs=['nvreader'],
    packages=find_packages("."),
    requires=[
        "numpy",
        "pillow",
        "cupy_cuda12x",
        "opencv_python_headless",
        "nvidia_dali_cuda120"
    ]
)