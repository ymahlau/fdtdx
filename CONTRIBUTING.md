# Contributing

We encourage community contributions of any kind to fdtdx! The following includes a list of useful information on how to make meaningful contributions.

## Installation

For development installation, make a fork of the repository, clone the fork and install in editable mode:

```bash
git clone https://github.com/link/to/your/fork/of/fdtdx
cd fdtdx
pip install -e .[dev]
git checkout -b name-of-your-feature-branch
```
If you have a graphics card or other accelerator you can also add the extras for these as described in the installation instructions. 

## Pull Requests

If you want to get some early feedback, it is very useful to make a draft pull request. This way we (the development team) can review the changes on a high-level early on.

Pull requests should follow the standard guidelines for good software development:
- The changes of a pull-request should adress a single feature or issue. Please do not make pull-requests with various changes at once. In that case split the PR into multiple individual PRs.
- In line with the point above, a pull-request should only contain a single commit with a meaningful commit message.

If you made multiple commits during development, you can squash these together using the following commands
```bash
git reset --soft HEAD~3  # squash the last three commits
git commit -m "new commit message"
git push -f
```

## Preventing merge conflicts

To prevent merge-conflicts, we recommend keeping your fork up to date with the fdtdx repo regularly. You can do this by first registering the original repo as a remote to your fork:
```bash
git remote add upstream https://github.com/ymahlau/fdtdx.git
```

Afterwards, you can regularly pull the latest changes from the main branch:
```bash
git pull upstream main
```

## Code quality

We have automatic checks installed which keep the code formatting of fdtdx in a (mostly) standardized way. You can run the pre-commit pipeline to check (and automatically fix) your formatting:
```bash
pre-commit run --all
```

## Questions

If you have any questions do not hesitate to ask them! We know that it can be very challenging to get started with JAX specifically and will help you with all problems that might come up.

If there exists already an issue or discussion regarding the feature which you would like to implement, please post your question there. If there does not exist an issue / discussion, create a new one!

