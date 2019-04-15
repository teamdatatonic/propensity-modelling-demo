from setuptools import find_packages
from setuptools import setup

# package meta-data
NAME = 'mlengine'
DESCRIPTION = 'Propensity modelling in TensorFlow for large retailer.'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = 0.1

# dependencies
REQUIRED = [
    'argparse>=1.4.0', 'numpy>=1.15.4', 'pandas>=0.23.4',
    'scikit-learn>=0.20.1', 'dask>=1.0.0', 'tensorflow>=1.12.0',
    'protobuf>=3.6.1', 'toolz>=0.7.3', 'partd>=0.3.8',
    'cloudpickle>=0.2.1', 'gcsfs>=0.2.0'
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True)
