"""Packaging metadata for the JAXVacua distribution.

Purpose
-------
Declare the package name, version, project metadata and installation
dependencies used by setuptools.

Main public API
---------------
- ``setup(...)`` invocation for building and installing ``jaxvacua``.

Design notes
------------
Runtime package behaviour lives in ``jaxvacua``.  Keep this file focused on
packaging metadata and avoid importing heavy project modules here.
"""

from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='jaxvacua',
    version='0.1.2',
    description='A JAX-based framework for sampling and analysing flux vacua in string theory.',
    long_description=Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Andreas Schachner',
    author_email='as3475@cornell.edu',
    url='https://github.com/StringJAX/jaxvacua',
    license='GPL-3.0-or-later',
    # find_packages() picks up jaxvacua AND its subpackages (jaxvacua.conifold,
    # ...).  The old packages=['jaxvacua'] shipped a broken wheel that failed at
    # import with `ModuleNotFoundError: No module named 'jaxvacua.conifold'`.
    packages=find_packages(exclude=['tests', 'tests.*']),
    # Ship the bundled Kreuzer-Skarke model files (jaxvacua/models/h12_<N>/
    # model_<M>.p) so `FluxVacuaFinder(h12=..., model_ID=...)` works straight
    # from a PyPI install.  ``models/*/*.p`` matches the two-level layout
    # exactly (and excludes any ``.ipynb_checkpoints`` a recursive glob would
    # otherwise sweep in).
    package_data={'jaxvacua': ['models/*/*.p']},
    python_requires='>=3.12',
    install_requires=[
        'numpy',
        'scipy',
        'jax',
        'jaxlib',
        'optax',
        'pandas',
        'tqdm',
        'python-flint',
        'gurobipy',
        'cytools',
        'jaxpolylog>=0.3.0',
    ],
    extras_require={
        'notebooks': ['jupyterlab', 'ipywidgets', 'anywidget'],
        'viz': ['matplotlib', 'seaborn', 'plotly'],
    },
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
    ],
)
