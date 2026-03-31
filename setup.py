"""
Build script for the C++ pybind11 solver module.
Uses MSVC (cl.exe) via setuptools.
"""
from setuptools import setup, Extension
import pybind11

cpp_solver = Extension(
    'cpp_solver',
    sources=['src/optimizer/cpp/solver.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['/std:c++17', '/O2', '/EHsc', '/openmp'],
)

setup(
    name='amie_apo_cpp_solver',
    version='1.0.0',
    description='AMIE-APO C++ Portfolio Optimization Kernels',
    ext_modules=[cpp_solver],
)
