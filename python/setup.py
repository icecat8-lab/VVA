from setuptools import setup, find_packages

setup(
    name="vva-framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "vva-cli=vva.cli:main",
        ],
    },
)
