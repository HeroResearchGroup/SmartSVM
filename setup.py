#!/usr/bin/env python

import os
import re

#from distutils.core import setup
from setuptools import setup, find_packages
from distutils.extension import Extension

# Set this to True to enable building extensions using Cython. Set it to False 
# to build extensions from the C file (that was previously generated using 
# Cython). Set it to 'auto' to build with Cython if available, otherwise from 
# the C file.
USE_CYTHON = 'auto'

if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        if USE_CYTHON == 'auto':
            USE_CYTHON = False
        else:
            raise

cmdclass = {}
ext_modules = []

if USE_CYTHON:
    ext_modules += [
            Extension("smartsvm.ortho_mst", [
                    os.path.join("src", "ortho_mst.pyx"),
                    os.path.join("src", "c_ortho_mst.c")
                    ]),
            Extension("smartsvm.multiclass_mst_count", [
                os.path.join("src", "multiclass_mst_count.pyx"),
                os.path.join("src", "c_multiclass_mst_count.c")
                ])
            ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
            Extension("smartsvm.ortho_mst", [
                os.path.join("src", "ortho_mst.c"),
                os.path.join("src", "c_ortho_mst.c")
                ]),
            Extension("smartsvm.multiclass_mst_count", [
                os.path.join("src", "multiclass_mst_count.c"),
                os.path.join("src", "c_multiclass_mst_count.c")
                ])
            ]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


version = re.search("__version__ = '([^']+)'", 
        open('smartsvm/__init__.py').read()).group(1)

setup(
        name='smartsvm',
        version=version,
        description=("Python package for Meta-Learning and Adaptive "
            "Hierarchical Classifier Design"),
        long_description=read('README.rst'),
        url="https://github.com/HeroResearchGroup/SmartSVM",
        author='G.J.J. van den Burg',
        author_email='gertjanvandenburg@gmail.com',
        license='GPL v2',
        install_requires=[
            'networkx>=1.9',
            'scipy',
            'scikit-learn',
            'numpy'
            ],
        packages=find_packages(),
        cmdclass=cmdclass,
        ext_modules=ext_modules,
        classifiers=[
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            ]
)
