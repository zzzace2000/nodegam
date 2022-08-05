# setup.py

from setuptools import setup, find_packages
from pathlib import Path

name = "nodegam"
version = "0.3.0"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=name,
    version=version,
    author="Chun-Hao Chang",
    author_email="chkchang21@gmail.com",
    description="NodeGAM - an interpretable deep learning GAM model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zzzace2000/nodegam",
    packages=find_packages(),
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch>=1.1.0",
        "numpy>=0.13",
        "scipy>=1.2.0",
        "scikit-learn>=0.17",
        "catboost>=0.12.2",
        "xgboost>=0.81",
        "matplotlib",
        "tqdm",
        "tensorboardX",
        "pandas",
        "prefetch_generator",
        "requests",
        "category_encoders",
        "filelock",
        "qhoptim",
        "mat4py",
        "interpret>=0.2",
        "pygam",
        "seaborn",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)