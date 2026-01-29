"""
Setup script for MAB Debiasing System
"""

from setuptools import setup, find_packages

setup(
    name="mab_debiasing",
    version="0.1.0",
    description="Adaptive Multi-Armed Bandit Debiasing Strategy Selection for Multilingual LLMs",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "sentence-transformers>=2.2.0",
        "langdetect>=1.0.9",
        "fasttext-wheel>=0.9.2",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "fire>=0.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
