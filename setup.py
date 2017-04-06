from setuptools import setup

setup(name = 'vampyre',
      version = '0.0',
      description = 'Vampyre is a Python package for generalized Approximate Message Passing',
      author = 'GAMP Team',
      setup_requires=['pytest','pytest-runner'],
      install_requires = ['pytest-runner','numpy','scipy','matplotlib'],
      tests_require = ['pytest','numpy','scipy'],
      author_email = 'gampteam@gmail.com',
      license = 'MIT',
      packages = ['vampyre'],
      zip_safe = False)