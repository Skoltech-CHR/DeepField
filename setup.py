"""A framework for reservoir simulation."""
import re
from setuptools import setup, find_packages

with open('deepfield/__init__.py', 'r') as f:
    VERSION = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if not VERSION:
        raise RuntimeError("Unable to find version string.")
    VERSION = VERSION.group(1)

with open('docs/index.rst', 'r') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='DeepField',
    packages=find_packages(exclude=['docs']),
    version=VERSION,
    url='https://github.com/Skoltech-CHR/DeepField',
    license='Apache License 2.0',
    author='Skoltech-CHR',
    author_email='',
    description='A framework for reservoir simulation',
    long_description=LONG_DESCRIPTION,
    zip_safe=False,
    platforms='any',
    install_requires=[
        'torchdiffeq',
        'vtk==9.0.1',
        'torchvision',
        'torch',
        'numpy>=1.15.1',
        'h5py>=2.8.0',
        'scikit-image>=0.14.0',
        'scipy>=1.1.0',
        'ipywidgets>=7.4.1',
        'anytree>=2.7.2',
        'pandas>=0.25.3',
        'numba>=0.39.0',
        'pyvista==0.23.1',
        'matplotlib',
        'PyQt5==5.11.3',
        'panel',
        'chardet',
        'scikit-learn>=0.21.2',
        'pytest==3.8.0',
        'psutil>=5.6.5',
        'deprecated==1.2.9',
        'tables==3.6.1'
    ],
    extras_require={
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Apache License 2.0',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
)
