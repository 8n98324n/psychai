# setup.py

from setuptools import setup, find_packages

setup(
    name="psychai",
    version="0.1",
    packages=find_packages(),
    install_requires=[],  # Add any dependencies here
    author="Ivan Liu",
    author_email="ivanliu@bnu.edu.cn",
    description="AI Toolbox for Psychological and Behavioral Research",
    url="https://https://github.com/8n98324n/psychai",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
