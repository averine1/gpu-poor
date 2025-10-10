from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gpu-poor",
    version="3.0.0",  
    author="Averine",
    author_email= "averine1556@outlook.com",  
    description="Extreme LLM compression (50-79%) for CPU inference with zero quality loss",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/averine1/gpu-poor",  
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
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.19.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "benchmark": [
            "psutil>=5.9.0",  
            "matplotlib>=3.5.0",
        ],
    },
    keywords="llm quantization compression cpu-inference transformers pytorch int4 int8",
    project_urls={
        "Bug Reports": "https://github.com/averine1/gpu-poor/issues",
        "Source": "https://github.com/averine1/gpu-poor",
    },
)