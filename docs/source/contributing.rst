Contributing
============

We encourage community contributions of any kind to fdtdx! The following includes a list of useful information on how to make meaningful contributions.

Installation
------------

For development installation, make a fork of the repository, clone the fork and install in editable mode:

.. code-block:: bash

   git clone https://github.com/link/to/your/fork/of/fdtdx
   cd fdtdx
   pip install -e .[dev]
   git checkout -b name-of-your-feature-branch

If you have a graphics card or other accelerator you can also add the extras for these as described in the installation instructions.

If you want to also install a different version of jax (for example with cuda support for GPU-acceleration) you can do so at any time after installing fdtdx. Simply run ``pip install jax[cuda] -U`` in your development environment.

You can view the detailed list of dependencies installed for every specified configuration in the ``pyproject.toml`` file.

Creating and running unit tests
--------------------------------

Testing is an important part of software development. Therefore, we ask you to create unit tests for any software that you write for this repository. Please also run all unit tests before starting pull request to make sure that all existing test cases still work as intended. 

You can run the unit tests with the ``pytest`` command in the fdtdx repository. If you want to specify a single test file to run you could specify the file after the command itself: ``pytest tests/conversion/test_export`` 

The unit tests are located in the ``/tests`` folder, which mirrors the structure of the ``/src`` folder. Therefore, if you are adding software to a file in the /src folder, please add the test cases at the corresponding location in the /test folder.

Pull Requests
-------------

If you want to get some early feedback, it is very useful to make a draft pull request. This way we (the development team) can review the changes on a high-level early on.

Pull requests should follow the standard guidelines for good software development:

- The changes of a pull-request should adress a single feature or issue. Please do not make pull-requests with various changes at once. In that case split the PR into multiple individual PRs.
- In line with the point above, a pull-request should only contain a single commit with a meaningful commit message.

If you made multiple commits during development, you can squash these together using the following commands

.. code-block:: bash

   git reset --soft HEAD~3  # squash the last three commits
   git commit -m "new commit message"
   git push -f

Preventing merge conflicts
---------------------------

To prevent merge-conflicts, we recommend keeping your fork up to date with the fdtdx repo regularly. You can do this by first registering the original repo as a remote to your fork:

.. code-block:: bash

   git remote add upstream https://github.com/ymahlau/fdtdx.git

Afterwards, you can regularly pull the latest changes from the main branch:

.. code-block:: bash

   git pull upstream main

Code quality
------------

We have automatic checks installed which keep the code formatting of fdtdx in a (mostly) standardized way. You can run the pre-commit pipeline to check (and automatically fix) your formatting:

.. code-block:: bash

   pre-commit run --all

Questions
---------

If you have any questions do not hesitate to ask them! We know that it can be very challenging to get started with JAX specifically and will help you with all problems that might come up.

If there exists already an issue or discussion regarding the feature which you would like to implement, please post your question there. If there does not exist an issue / discussion, create a new one!