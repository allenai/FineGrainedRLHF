from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(
    name="fgrlhf",
    version="0.1.0",
    description="A library for fine-grained rlhf",
    author="Zeqiu Wu*, Yushi Hu*",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    url="https://github.com/allenai/FineGrainedRLHF",
)
