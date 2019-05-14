from distutils.core import setup, Command
from setuptools import find_packages

setup(
    name = "smrt",
    packages = find_packages(exclude='test'),
    version = "0.9",
    description = "Snow Microwave Radiative Transfer model",
    author = "Ghislain Picard, Melody Sandells, Henning Loewe",
    author_email = "ghislain.picard@univ-grenoble-alpes.fr, melody.sandells@gmail.com, loewe@slf.ch",
    url = "https://github.com/smrt-model/smrt",
    keywords = ["radiative transfer","model","snow","microwave"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        ],
    long_description = """\
The Snow Microwave Radiative Transfer (SMRT) model is a highly modular model to compute the thermal emission of snowpacks and other cryospheric bodies in the microwave domain.

SMRT is compatible with Python 3.5+
"""
)
