from setuptools import setup, find_packages

setup(name='neurotools',
      version='0.2',
      description='General helper functions for working with neuroimaging data.',
      url='http://github.com/sahahn/neurotools',
      author='Sage Hahn',
      author_email='sahahn@uvm.edu',
      license='MIT',
      python_requires=">=3.7",
      packages=find_packages(),
      install_requires=[
          'scikit-learn>=0.24.0',
          'scipy',
          'numpy',
          'pandas>=1.1',
          'nibabel>=3.2.1',
          'joblib',
          'statsmodels>=0.12',
          'matplotlib>=3.3.4',
          'nilearn>=0.8.1',
          'networkx',
          'neuromaps @ https://github.com/netneurolab/neuromaps/archive/refs/tags/0.0.1.zip',
          'neurotools_data @ https://github.com/sahahn/neurotools_data/archive/refs/tags/1.2.zip'
      ],
      test_suite='pytest',
      tests_require=['pytest', 'coverage'],
      zip_safe=False)
