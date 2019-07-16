from setuptools import find_packages
from setuptools import setup

# package meta-data
NAME = 'ai-platform'
DESCRIPTION = 'Propensity modelling in TensorFlow for large retailer.'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = 0.1

# dependencies
REQUIRED = [
    'argparse>=1.4.0', 'tensorflow>=1.13.1',
    'protobuf>=3.6.1', 'gcsfs>=0.2.0'
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True)
