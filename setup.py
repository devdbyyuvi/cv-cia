from setuptools import setup, find_packages

setup(
    name="neural-relighting",
    version="0.1.0",
    description="Feed-forward inverse rendering (geometry+BRDF+lighting) from sparse casual captures",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
)
