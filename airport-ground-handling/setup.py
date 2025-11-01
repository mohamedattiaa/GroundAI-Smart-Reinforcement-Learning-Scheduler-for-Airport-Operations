from setuptools import setup, find_packages

setup(
    name="airport-ground-handling",
    version="1.0.0",
    description="Synthetic dataset generator for airport ground handling operations",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scipy>=1.10.0',
        'pyyaml>=6.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'tqdm>=4.65.0',
        'click>=8.1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.1.0',
            'jupyter>=1.0.0',
        ],
        'ml': [
            'scikit-learn>=1.2.0',
            'sentence-transformers>=2.2.0',
            'chromadb>=0.4.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'generate-airport-data=generate_dataset:main',
        ],
    },
    python_requires='>=3.8',
)