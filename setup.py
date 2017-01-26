from setuptools import setup

setup(name = 'vampyre',
      version = '0.0',
      description = 'Vampyre is a Python package for generalized Approximate Message Passing',
      author = 'GAMP Team',
      install_requires = ['numpy','scipy','matplotlib'],
      test_suite = 'nose.collector',
      test_require = ['nose','nose-timer','nose-ignore-docstring','numpy','scipy'],
      author_email = 'gampteam@gmail.com',
      license = 'MIT',
      packages = ['vampyre'],
      zip_safe = False)