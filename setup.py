from setuptools import setup, find_packages

def read_requirements(fname):
    """Read and parse the requirements.txt file."""
    with open(fname) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="protein2pam",
    version="0.1.0",
    description="Deep learning prediction of CRISPR-Cas PAMs",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "protein2pam=protein2pam.cli:main",
        ],
    },
)
