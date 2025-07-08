from setuptools import setup, find_packages

setup(
    name="spatial_cluster_helper",
    version="0.1.3",
    author="Luc Anselin, Pedro Amaral",
    author_email="lanselin@gmail.com, pedrovma.ufmg@gmail.com",
    description="Support functions for spatial cluster analysis following Anselin (2024)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lanselin/notebooks_for_spatial_clustering",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "pandas",
        "geopandas",
        "scikit-learn", 
        "scipy",
        "matplotlib",
        "libpysal"
    ],
)