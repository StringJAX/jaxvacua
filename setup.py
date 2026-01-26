from setuptools import setup

setup(
    name='jaxvacua',
    version='0.0.1',
    description='',
    author='Andreas Schachner',
    author_email='andreas.schachner@gmx.de',
    packages=['jaxvacua'],
    python_requires='>=3.12',
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
	    'optax',
        'partial',
        'matplotlib',
        'seaborn',
        'jupyter',
        'h5py',
        'pandas',
        'tqdm',
        'sympy',
        'flax',
        'jaxpolylog@git+https://github.com/AndreasSchachner/jaxpolylog.git#egg=jaxpolylog'
    ],
)
