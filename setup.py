#!/usr/bin/env python


from setuptools import setup, find_packages
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

from simple_chess import __version__


def get_reqs():
    install_reqs = parse_requirements('requirements.txt')
    return [str(ir.req) for ir in install_reqs]


setup(
    name='simple_chess',
    version=__version__,
    description='A simple implementation for simple chess',
    author='Ary',
    author_email='mrgao.ary@gmail.com',
    license='MIT License',
    url='https://github.com/tokkiu/simple_chess',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'six>=1.9.0',
    ],
    extras_require={
        'dev': [
            'prospector>=0.10.2',
        ],
        'test': [
            'coverage>=3.7.1',
            'coveralls>=0.5',
            'nose>=1.3.7',
            'python-coveralls>=2.5.0',
        ],
        'docs': [
            'Sphinx>=1.3.1',
        ],
    },
    test_suite='nose.collector',
)
