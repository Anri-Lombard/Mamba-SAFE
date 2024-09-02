from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mamba-safe",
    version="1.0.1",
    author="Anri Lombard",
    author_email="anri.m.lombard@gmail.com",
    description="A framework to generate molecules with the mamba architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Anri-Lombard/DrugGPT",
    packages=find_packages(where="mamba_safe"),
    package_dir={"": "mamba_safe"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "transformers",
        "torch",
        "accelerate",
        "datasets",
        "mamba_ssm",
        "causal-conv1d"
    ],
)