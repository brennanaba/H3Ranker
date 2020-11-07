from setuptools import setup, find_packages

setup(
    name='H3Ranker',
    version='0.0.1',
    description='Set of useful functions to develop a CDRH3 scoring function.',
    license='BSD 3-clause license',
    include_package_data=True,
    packages = find_packages(include=('H3Ranker', 'H3Ranker.*')),
)
