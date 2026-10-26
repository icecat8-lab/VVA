from setuptools import find_packages, setup

setup(
    name="vva-converter",
    version="1.1.0",
    packages=find_packages(),
    install_requires=["numpy>=1.19.0"],
    extras_require={"torch": ["torch>=1.8.0"]},
    entry_points={
        "console_scripts": [
            "vva-convert=vva.cli:main",
        ],
    },
    author="Icecat",
    description="Converts RRDBNet super-resolution checkpoints to the .vva format used by the VVA engine.",
    python_requires=">=3.8",
)
