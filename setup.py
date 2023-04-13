from setuptools import setup, find_packages
from Cython.Build import cythonize
from os.path import splitext, basename
from glob import glob
import numpy

setup(
    name='tobac flow',
    version='1.7.3',
    description="Detection and tracking of deep convective clouds in high time resolution geostationary satellite imagery",
    url='',
    author='William Jones',
    author_email='william.jones@physics.ox.ac.uk',
    license='BSD-3',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    install_requires=[],
    ext_modules=cythonize("./src/tobac_flow/_watershed.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
