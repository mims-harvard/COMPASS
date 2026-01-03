import sys
import os
from setuptools import setup, find_packages
from setuptools import Command
from setuptools.command.test import test as TestCommand
from datetime import datetime
import compass

def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]


with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

NAME = "immuno-compass"
VERSION = compass.__version__
AUTHOR = "WanXiang Shen"
DESCRIPTION = "COMPASS: Generalizable AI predicts immunotherapy outcomes across cancers and treatments."
URL = "https://github.com/mims-harvard/COMPASS/tree/main"

REQUIRED_PYTHON_VERSION = (3, 8)
PACKAGES = find_packages(exclude = ['test', 'gallery', 'misc', 'example', '.ipynb_checkpoints',])

INSTALL_DEPENDENCIES = parse_requirements('./requirements.txt')
SETUP_DEPENDENCIES = []
TEST_DEPENDENCIES = ["pytest"]
EXTRA_DEPENDENCIES = {"dev": ["pytest"]}


    
if sys.version_info < REQUIRED_PYTHON_VERSION:
    sys.exit("Python >= 3.8 is required. Your version:\n" + sys.version)


class PyTest(TestCommand):
    """
    Use pytest to run tests
    """

    user_options = [("pytest-args=", "a", "Arguments to pass into py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name=NAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    version=VERSION,
    author=AUTHOR,
    packages=PACKAGES,
    include_package_data=True,
    #package_data = PACKAGE_DATA,
    install_requires=INSTALL_DEPENDENCIES,
    setup_requires=SETUP_DEPENDENCIES,
    tests_require=TEST_DEPENDENCIES,
    extras_require=EXTRA_DEPENDENCIES,
    cmdclass={"test": PyTest},
)