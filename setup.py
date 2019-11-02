"""Install package."""
from setuptools import setup, find_packages

setup(
    name='paccmann_omics',
    version='0.0.1',
    description='Generative models of omic data for PaccMann^RL',
    long_description=open('README.md').read(),
    url='https://github.com/PaccMann/paccmann_omics',
    author='Jannis Born, Ali Oskooei, Matteo Manica, Joris Cadow',
    author_email=(
        'jab@zurich.ibm.com, ali.oskooei@gmail.com, '
        'drugilsberg@gmail.com, joriscadow@gmail.com'
    ),
    install_requires=["numpy", 'scipy', "torch", "tensorflow"],
    packages=find_packages('.'),
    zip_safe=False,
)
