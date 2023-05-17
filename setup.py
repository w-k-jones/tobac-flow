from setuptools import setup, find_packages
import pathlib
import numpy as np
from Cython.Build import cythonize

cwd = pathlib.Path("./")
src = cwd / "src"
packages = find_packages(src)
modules = sorted(list(src.glob("**/[:alpha:]*.py")))
cython_modules = list(src.glob("**/*.pyx"))
setup(
    name="tobac-flow",
    version="1.7.4",
    description="Detection and tracking of deep convective clouds in high time resolution geostationary satellite imagery",
    url="",
    author="William Jones",
    author_email="william.jones@physics.ox.ac.uk",
    license="BSD-3",
    packages=packages,
    package_dir={"": "src"},
    py_modules=modules,
    include_package_data=True,
    install_requires=[],
    ext_modules=cythonize(cython_modules),
    include_dirs=[np.get_include()],
    zip_safe=False
)
