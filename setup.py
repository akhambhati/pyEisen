import io
import os
import re

from setuptools import find_packages, setup

import version

HERE = os.path.abspath(os.path.dirname(__file__))

# ==============================================================================
# Variables
# ==============================================================================

NAME = "pyEisen"
VERSION = version.get_version()
DESCRIPTION = "pyEisen: A No-Frills Repository of Time-Frequency Analysis Tools."
with open(os.path.join(HERE, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
PACKAGES = find_packages()
AUTHOR = "Ankit N. Khambhati"
AUTHOR_EMAIL = "akhambhati@gmail.com"
DOWNLOAD_URL = 'http://github.com/akhambhati/pyeisen/'
LICENSE = 'MIT'
INSTALL_REQUIRES = ['numpy', 'scipy']

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
)
