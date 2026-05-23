from setuptools import setup, find_packages

setup(
    name="qec_ml",
    version="0.1.0",
    description="Machine Learning for Quantum Error Correction",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "stim>=1.13",
        "pymatching>=2.1",
        "scipy>=1.11",
        "tqdm>=4.65",
        "pandas>=2.0",
    ],
    extras_require={
        "gnn": ["torch-geometric>=2.4"],
        "dev": ["pytest", "jupyter", "ipywidgets"],
    },
    python_requires=">=3.9",
)
