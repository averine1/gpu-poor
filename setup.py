from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gpu-poor",
    version="0.1.0",
    author="Your Name",
    description="Run AI models without a GPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gpu-poor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "psutil>=5.9.0",
    ],
)