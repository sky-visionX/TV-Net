# cython: language_level=3
#cython_2darr_setup.py
#fumiama 20201225

 
#添加 include_dirs=[np.get_include()]
 
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
 
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = cythonize("sliding_window.pyx"),
    include_dirs=[np.get_include()])

