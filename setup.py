#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

readme = ""
license = ""

setup(
    name = 'edit_nts',
    version = '0.0.1',
    description = 'reimplementation of editnts',
    long_description = readme,
    license = license,
    packages = find_packages(exclude = ('notes', 'docs', 'tests'))
)
