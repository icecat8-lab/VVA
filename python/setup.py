from setuptools import setup, find_packages

setup(
    name="vva-converter",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0"
    ],
    entry_points={
        "console_scripts": [
            "vva-convert=vva.cli:main",
        ],
    },
    author="Icecat",
    description="A lightweight model exporter utilizing selective mixed-precision for custom mobile inference engines.",
    python_requires=">=3.8",
)
