from setuptools import setup, find_packages

setup(
    name="metaquest",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "metaquest = metaquest:main",
        ]
    },
    install_requires=[
        "argparse",
        "os",
        "csv",
        "logging",
        "subprocess",
        "pathlib",
        "collections",
        "pandas",
        "Bio",
        "urllib",
        "lxml",
        "upsetplot",
        "matplotlib",
        "numpy",
    ],
)

