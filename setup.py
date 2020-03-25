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
    name='bc-selector',
    version=version,
    author='Tomasz Klonecki',
    author_email="tomasz.klonecki@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/Kaketo/bc-selector',
    install_requires=REQUIREMENTS,
    license='MIT',
    packages=find_packages('.'),
    zip_safe=True
)