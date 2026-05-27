from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "pysim._accelerators",
        ["src/pysim/_accelerators.cpp"],
        extra_compile_args=["-fopenmp", "-g", "-std=gnu++11"],
        extra_link_args=["-fopenmp", "-lpthread"],
    ),
]

setup(
    ext_modules=ext_modules,
    packages=["pysim"],
    package_dir={"": "src/"},
)
