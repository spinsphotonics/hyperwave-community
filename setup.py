from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hyperwave-community",
    version="0.1.0",
    author="Hyperwave Team",
    author_email="support@spinsphotonics.com",
    description="Open-source photonics simulation toolkit with GPU-accelerated FDTD via cloud API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spinsphotonics/hyperwave-community",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "requests>=2.26.0",
        "gdstk>=0.9.0",
        "gdsfactory>=7.0.0",
        "scikit-image>=0.19.0",
        "cloudpickle>=2.0.0",
    ],
    extras_require={
        "docs": [
            "sphinx>=8.0",
            "pydata-sphinx-theme",
        ],
    },
)
