"""
Smart Summarizer - Setup Configuration
Professional text summarization application with multiple AI models
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Smart Summarizer - AI-powered text summarization tool"

setup(
    name="smart-summarizer",
    version="1.0.0",
    author="Abdul Razzaq Ansari",
    author_email="rajakansari83@gmail.com",
    description="AI-powered text summarization with multiple model comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rajak13/Smart-Summarizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.1",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smart-summarizer=app.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Rajak13/Smart-Summarizer/issues",
        "Source": "https://github.com/Rajak13/Smart-Summarizer",
        "Documentation": "https://smart-summarizer.readthedocs.io/",
    },
)