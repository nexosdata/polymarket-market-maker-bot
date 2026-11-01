from setuptools import setup, find_packages

setup(
    name="polymarket-market-maker-bot",
    version="1.0.0",
    description="Institutional Avellaneda-Stoikov Market Making Engine for Polymarket & Kalshi Prediction Markets",
    author="NaxosData Quant Engineering & William Gonda",
    author_email="research@naxosdata.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22.0",
        "httpx>=0.24.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "polymarket-mm=market_maker.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
