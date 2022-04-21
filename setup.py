from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='bp-neurotools',
      version='0.22',
      long_description=long_description,
      long_description_content_type='text/markdown',
      description='General helper functions for working with neuroimaging data.',
      url='http://github.com/sahahn/neurotools',
      author='Sage Hahn',
      author_email='sahahn@uvm.edu',
      license='MIT',
      python_requires=">=3.7",
      packages=find_packages(),
      install_requires=[
          'scikit-learn>=1',
          'scipy',
          'numpy',
          'pandas>=1.1',
          'nibabel>=3.2.1',
          'joblib',
          'statsmodels>=0.13.2',
          'matplotlib>=3.3.4',
          'nilearn>=0.8.1',
          'networkx'
      ],
      test_suite='pytest',
      tests_require=['pytest', 'coverage'],
      zip_safe=False)
