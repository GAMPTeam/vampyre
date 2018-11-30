from setuptools import setup

import numpy as np
np.random.seed(1) # make the tests repeatable

setup(name = 'vampyre',
      version = '0.0',
      description = 'Vampyre is a Python package for generalized Approximate Message Passing',
      author = 'GAMP Team',
      install_requires = ['nose','nose-timer','numpy','scipy','matplotlib','pywavelets','scikit-learn',],
      test_suite = 'nose.collector',
      tests_require = ['nose','nose-timer','numpy','scipy','PyWavelets'],
      author_email = 'gampteam@gmail.com',
      license = 'MIT',
      packages = ['vampyre'],
      zip_safe = False)
