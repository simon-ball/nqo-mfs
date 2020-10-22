#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

with open('README.md') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split()



setup(
    author="Simon Ball",
    author_email='s.w.ball@st-aidans.com',
    classifiers=[
        'Development Status :: 5 - release',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Magnetic field visualisation code from the NQO group at SDU.",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='physics magnetism magnets',
    name='mfs',
    packages=find_packages(include=['mfs*']),
    url='https://github.com/simon-ball/nqo-mfs',
    version='1.1.0',
    zip_safe=False,
)