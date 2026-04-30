"""Setup for cli-anything-lfiax: CLI harness for Likelihood Free Inference in JAX."""
from setuptools import setup, find_namespace_packages

setup(
    name="cli-anything-lfiax",
    version="1.0.0",
    description="CLI harness for LFIAX — Likelihood Free Inference in JAX",
    long_description=open("cli_anything/lfiax/README.md").read(),
    long_description_content_type="text/markdown",
    author="cli-anything",
    python_requires=">=3.9",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    package_data={
        "cli_anything.lfiax": ["skills/*.md"],
        "cli_anything.lfiax.examples": ["*.json", "*.py"],
    },
    include_package_data=True,
    install_requires=[
        "click>=8.0",
        "pyyaml>=5.0",
        "prompt_toolkit>=3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "cli-anything-lfiax=cli_anything.lfiax.lfiax_cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
    ],
)
