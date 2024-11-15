from setuptools import setup, find_packages

# Build configuration for automating tasks
setup(
    name="Salary_Prediction",  # Project name
    version="1.0.0",  # Project version for version tracking
    description="A project to predict salaries based on demographic data",
    author="Mehdi Nemri",
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[  # Dependency management
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "matplotlib>=3.2.0",
        "seaborn>=0.10.0",
        "scikit-learn>=0.22.0"
    ],
    classifiers=[ 
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
