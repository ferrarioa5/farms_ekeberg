
from setuptools import setup, find_packages
import numpy as np

with open("README.md", "r") as fh:
    description = fh.read()

setup(
    name="farms_ekeberg",
    author="Andrea Ferrario",
    description="Implicit muscles simulation library",
    packages=find_packages(),
    long_description=description,
    include_package_data=True,
    include_dirs=[np.get_include(), 'src'],
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    zip_safe=False,
)