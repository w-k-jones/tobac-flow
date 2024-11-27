from setuptools import setup, find_packages
import pathlib
import numpy as np
from Cython.Build import cythonize

src = pathlib.Path(".") / "tobac_flow"
packages = ["tobac_flow"] + [
    f"tobac_flow/{package}" for package in find_packages(str(src))
]
modules = sorted([str(f.relative_to(f.parts[0])) for f in src.rglob("**/[a-z]*.py")])
cython_modules = [str(f) for f in src.rglob("*.pyx")]

def read(pkg_name):
    init_fname = pathlib.Path(__file__).parent / pkg_name / "__init__.py"
    with open(init_fname, "r") as fp:
        return fp.read()

def get_version(pkg_name):
    for line in read(pkg_name).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="tobac-flow",
    version=get_version("tobac_flow"),
    description="Detection and tracking of deep convective clouds in high time resolution geostationary satellite imagery",
    url="",
    author="William Jones",
    author_email="william.jones@physics.ox.ac.uk",
    license="BSD-3",
    packages=packages,
    include_package_data=True,
    install_requires=[],
    ext_modules=cythonize(cython_modules),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
