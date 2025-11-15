"""
Setup script for Plastic Neural Networks package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "Plastic Neural Networks: Learning Through Iterative Delta Refinement"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith('#')
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
    ]

setup(
    name="plastic-neural-networks",
    version="1.0.0",
    author="Seungho Choi",
    author_email="madst0614@gmail.com",
    description="Recurrent delta refinement for efficient language modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/plastic-neural-networks",
    project_urls={
        "Paper": "https://doi.org/10.5281/zenodo.17548176",
        "Bug Tracker": "https://github.com/yourusername/plastic-neural-networks/issues",
    },
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=8.14.0",
            "ipywidgets>=8.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pnn-train=scripts.train:main",
            "pnn-evaluate=scripts.evaluate:main",
        ],
    },
)
