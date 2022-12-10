from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(name='tobac flow',
      version='1.6',
      description="""Detection and tracking of deep convective clouds in high
        time resolution geostationary satellite imagery""",
      url='',
      author='William Jones',
      author_email='william.jones@physics.ox.ac.uk',
      license='BSD-3',
      packages=find_packages(),
      install_requires=[],
      ext_modules=cythonize("./tobac_flow/_watershed.pyx"),
      include_dirs=[numpy.get_include()],
      zip_safe=False)
