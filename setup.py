import os
import setuptools

with open(os.path.join(os.path.dirname(__file__),"SpykeTorch/VERSION")) as f:
    version = f.read().strip() 


with open("README.md", "r") as fh:
    long_description = fh.read()

def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return requirements

    
setuptools.setup(
    name="SpykeTorch-miladmozafari", # Replace with your own username
    version=version,
    author="miladmozafari",
    author_email="",
    description="High-speed simulator of convolutional spiking neural networks with at most one spike per neuron.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", # write github url,
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
    python_requires='>=3.6',
)