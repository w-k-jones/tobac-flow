from setuptools import setup, find_packages
import pathlib
import numpy as np
from Cython.Build import cythonize

cwd = pathlib.Path("./")
src = cwd / "tobac_flow"
packages = ["tobac_flow"] + [
    f"tobac_flow/{package}" for package in find_packages(str(src))
]
modules = sorted([str(f.relative_to(f.parts[0])) for f in src.glob("**/[a-z]*.py")])
cython_modules = [str(f) for f in src.glob("*.pyx")]

setup(
    name="tobac-flow",
    version="1.8.2",
    description="Detection and tracking of deep convective clouds in high time resolution geostationary satellite imagery",
    url="",
    author="William Jones",
    author_email="william.jones@physics.ox.ac.uk",
    license="BSD-3",
    packages=packages,
    # package_dir={"": "tobac_flow"},
    # py_modules=modules,
    include_package_data=True,
    install_requires=[],
    ext_modules=cythonize(cython_modules),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
