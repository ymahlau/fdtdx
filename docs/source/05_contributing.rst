Contributing
============

We encourage community contributions of any kind to fdtdx! The following includes a list of useful information on how to make meaningful contributions.

Installation
------------

**1. Fork and Clone**
As the first step, make a fork of the fdtdx repository, clone the fork, and create a new branch for the feature you want to develop:

.. code-block:: bash

   git clone https://github.com/YOUR-USERNAME/fdtdx
   cd fdtdx
   git checkout -b name-of-your-feature-branch

**2. Set up the Environment (Recommended: uv)**
We recommend using `uv <https://docs.astral.sh/uv/>`__ for development speed and reliability.
First, follow the installation instructions on their `website <https://docs.astral.sh/uv/getting-started/installation/>`__.

Then, install the development dependencies:

.. code-block:: bash

   uv sync --extra=dev
   source .venv/bin/activate

This activates your virtual environment. You should run the activation command anytime you start a new shell.

*Note: If you need a specific version of JAX (e.g., with CUDA support), you can install it inside the environment:*

.. code-block:: bash

   pip install -U jax[cuda]

**Alternative: Standard pip**
If you prefer not to use `uv`, you can use a standard virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]

Checklist
---------
This is a checklist that you should go through before making a PR. The individual points are explained in more detail below.

* [ ] Implement your feature and unit tests for your new feature
* [ ] Check if all unit tests run: ``pytest tests/``
* [ ] Check if all pre-commit checks are passing: ``uv run pre-commit run --all``
* [ ] Add an entry of your changes / fixes / new features to the ``changelog.md`` file
* [ ] If you added any new features, that should be top-level imported, add them to ``__all__`` in ``src/fdtdx/__init__.py``
* [ ] If you want your new feature to be present in the documentation, add an entry to ``docs/source/api.rst``
* [ ] Optional: If you want to go above and beyond, add a notebook showcasing your feature in the `notebooks repository <https://github.com/ymahlau/fdtdx-notebooks/>`__. Afterwards you can add an entry in the docs to include your new notebook.


Creating and running unit tests
--------------------------------

Testing is an important part of software development. Therefore, we ask you to create unit tests for any software that you write for this repository. Please also run all unit tests before opening a pull request to make sure that all existing test cases still work as intended. 

You can run the unit tests with the ``pytest`` command in the fdtdx repository. If you want to specify a single test file to run you could specify the file after the command itself: ``pytest tests/conversion/test_export.py`` 

The unit tests are located in the ``/tests`` folder, which mirrors the structure of the ``/src`` folder. Therefore, if you are adding software to a file in the /src folder, please add the test cases at the corresponding location in the /test folder.

Pull Requests
-------------

If you want to get some early feedback, it is very useful to make a draft pull request. This way we (the development team) can review the changes on a high-level early on.

Pull requests should follow the standard guidelines for good software development:

- The changes of a pull-request should address a single feature or issue. Please do not make pull-requests with various changes at once. In that case split the PR into multiple individual PRs.


Preventing merge conflicts
--------------------------

To prevent conflicts, keep your fork synchronized with the main repository.

1.  **Register Upstream:**

    .. code-block:: bash

       git remote add upstream https://github.com/ymahlau/fdtdx.git

2.  **Sync Regularly:**
    Before starting new work, pull the latest changes:

    .. code-block:: bash

       git fetch upstream
       git merge upstream/main

Code quality
------------

We use automatic checks to standardize code formatting. You can install these checks to run automatically on every commit:

.. code-block:: bash

   pre-commit install

You can also run them manually at any time:

.. code-block:: bash

   pre-commit run --all-files

Documentation
-------------
You can locally build the sphinx documentation of fdtdx using the following commands.

.. code-block:: bash

   sh docs/scripts/sync_notebooks.sh && uv run sphinx-autobuild -W --keep-going docs/source/ docs/build/

If you changed anything in the documentation, make sure to run the command above to check if the documentation still works as expected.

Questions
---------

If you have any questions do not hesitate to ask them! We know that it can be very challenging to get started with JAX specifically and will help you with all problems that might come up.

If there exists already an issue or discussion regarding the feature which you would like to implement, please post your question there. If there does not exist an issue / discussion, create a new one!