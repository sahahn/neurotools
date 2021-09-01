from setuptools import setup, find_packages

setup(name='neurotools',
      version='0.1',
      description='General helper functions for working with neuroimaging data.',
      url='http://github.com/sahahn/neurotools',
      author='Sage Hahn',
      author_email='sahahn@uvm.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'scikit-learn',
          'scipy',
          'numpy',
          'pandas',
          'nibabel',
          'joblib',
          'statsmodels'
      ])