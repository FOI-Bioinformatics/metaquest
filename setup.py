from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from metaquest/__init__.py
with open(os.path.join("metaquest", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.1.0"

setup(
    name="metaquest",
    version=version,
    author="Andreas Sjödin",
    author_email="andreas.sjodin@gmail.com",
    description="A toolkit for analyzing metagenomic datasets based on genome containment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FOI-Bioinformatics/metaquest",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
    install_requires=[
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
        'biopython>=1.79',
        'lxml>=4.6.0',
        'upsetplot>=0.6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'flake8>=3.9.0',
            'black>=21.5b0',
            'mypy>=0.812',
        ],
    },
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'metaquest=metaquest.cli.main:main',
        ],
    },
    package_data={
        'metaquest': ['py.typed'],
    },
)