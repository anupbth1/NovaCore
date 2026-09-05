from setuptools import setup, find_packages

setup(
    name="novacore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "novacore=cli.main:cli",
        ],
    },
    author="NovaCore Team",
    description="Training-Free LLM - Encode datasets to weights without training",
    python_requires=">=3.8",
)
