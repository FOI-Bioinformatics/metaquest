from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="metaquest",
    version="0.1",
    author="Andreas Sjödin",
    author_email="andreas.sjodin@gmail.com",
    description="A package to analyze SRA datasets and identify containment of specified genomes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FOI-Bioinformatics/metaquest",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy',
        'umap-learn',
        'scikit-learn',
        'biopython',
        'upsetplot',
        'lxml',
        # other dependencies
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'metaquest=metaquest.metaquest:main',
        ],
    },
)

