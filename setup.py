from setuptools import setup, find_packages

setup(
    name="neurotrace",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
        "faiss-cpu>=1.7.0",  # or faiss-gpu
    ],
    python_requires=">=3.8",
    description="Neural state analysis and interpretability for LLMs",
    author="Your Name",
)
