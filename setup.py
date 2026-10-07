"""Setup script for medical text simplification package."""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="simplifier",
    version="0.1.0",
    description="Medical text simplification for clinical RAG applications",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)