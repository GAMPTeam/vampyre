# vampyre
## Approximate Message Passing in Python
This repository is a stable and (ideally) well-engineered object-oriented implementation of approximate message passing (AMP) in general settings. 

## Features

## Installation 
### OSX & *Nix
Until the project is read for distribution, vampyre will not be registered on Pypi and available via `pip` without downloading the development repository first. In the meantime, developers can build the package by first cloning the repository.

```bash
> git clone https://github.com/GAMPTeam/vampyre.git
```

Subsequently, the package can be installed via `pip` with the `-e` directive to allow for live development of the package without re-installing `vampyre` to your `site-packages` directory.

```bash
> cd /path/to/vampyre
> pip install -e .  # or with the '--user' option to install it unprivileged
```

Voilà.

## Testing
Testing is accomplished via `nose` and `nose-timer`. To run the full-suite of package tests, run the following from the root vampyre directory.

```bash
> python setup.py test
```

To run tests with timing results, run the following command from the root vampyre directory.
```bash
> nosetests --with-timer
```

All test scripts can be found in the `./vampyre/vampyre/test` directory. All python scripts prefaced as `test_` will have their functions prefaced as `def test_*:` run.


## Building Documentation
To build documentation, you need to make sure to have a number of Sphinx packages installed. Specifically, the requirements can be installed via

```bash
> pip install sphinx sphinx-autobuild sphinx-rtd-theme
```

The documentation can be built in the `docs` directory via `Make` on OSX/*Nix, 
```bash
> cd /path/to/vampyre/docs
> make html
```
or via a batch file on Windows
```dos
C:\path\to\vampyre\docs make.bat
```

Note that if you add a new module, you will need to recreate the module index
via the command:
```bash
> sphinx-apidoc  -o . ..\vampyre\
```

Documentation can be found at `_build/index.html`. Documentation will be hosted at [Readthedocs.org](https://readthedocs.org) once the package goes live on Pypi.


## Contributors
This repository is maintained by the GAMP Team.

## References
(A list of papers goes here.)