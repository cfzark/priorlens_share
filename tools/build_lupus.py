"""Optional C++ build for a source checkout; core examples do not require it."""
from pathlib import Path
import os
from setuptools import setup, Extension
import numpy
import pybind11

root = Path(__file__).resolve().parents[1]
os.chdir(root)
setup(name='priorlens-fast-kernels', version='0.1.0',
      package_dir={'': 'src'}, packages=['prioreffect'],
      script_args=['build_ext','--inplace'],
      ext_modules=[Extension('prioreffect._lupus_fast',
         sources=['cpp/lupus_fast.cpp'],include_dirs=[numpy.get_include(),pybind11.get_include()],
         language='c++',extra_compile_args=['-O3','-std=c++17','-fopenmp'],
         extra_link_args=['-fopenmp'])])
