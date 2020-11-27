import codecs
import json
import pathlib
import re

from setuptools import find_packages, setup

PACKAGE_PATH = pathlib.Path(__file__).parent

def read(*paths):
    file = str(PACKAGE_PATH.joinpath(*paths).resolve())
    with codecs.open(file) as f:
        return f.read()

import bcselector
version = bcselector.__version__

with codecs.open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

DESCRIPTION = 'Python package to help you in variable selection.'
LONG_DESCRIPTION = read('README.rst')

setup(
    name='bcselector',
    version=version,
    author='Tomasz Klonecki',
    author_email="tomasz.klonecki@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/Kaketo/bcselector',
    install_requires=REQUIREMENTS,
    license='MIT',
    packages=find_packages('.'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    include_package_data=True
)