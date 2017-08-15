# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='ntee',
    version='0.0.1',
    author='Studio Ousia',
    author_email='ikuya@ousia.jp',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ntee=ntee.cli:cli'
        ]
    }
)
