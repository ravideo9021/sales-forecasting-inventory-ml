"""
Setup script for the Sales Forecasting & Inventory Optimization ML Project.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sales-forecasting-inventory-ml",
    version="1.0.0",
    author="ravideo9021",
    author_email="ravideo9021@gmail.com",
    description="Sales Forecasting & Inventory Optimization ML platform (RetailIQ dashboard)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ravideo9021/sales-forecasting-inventory-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
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
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sales-forecast=main:main",
            "sales-dashboard=app:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning forecasting inventory-optimization time-series retail analytics",
) 