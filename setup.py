from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent

requirements = (BASE_DIR / "requirements.txt").read_text().splitlines()

setup(
    name="nemotron-ops-commander",
    version="0.1.0",
    description="AI-powered incident response system using NVIDIA Nemotron",
    author="Nemotron Ops Commander",
    packages=find_packages(exclude=("tests", "data", "k8s", "benchmarks")),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[r for r in requirements if r and not r.startswith("#")],
)
