from setuptools import setup, find_packages

setup(
    name='H3Ranker',
    version='0.0.1',
    description='Set of useful functions to develop a CDRH3 scoring function.',
    license='BSD 3-clause license',
    include_package_data=True,
    packages = find_packages(include=('H3Ranker', 'H3Ranker.*')),
    install_requires=[	
        'tensorflow>=2.0.0',	
        'biopython>=1.78',	
        'Keras>=2.3.0',	
        'numba>=0.51.2',	
        'numpy>=1.18.5',	
        'pandas>=1.1.3',	
    ],
)
