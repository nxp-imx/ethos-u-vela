# Vela Testing

## Tools

Vela's Python codebase is PEP8 compliant with the exception of a 120 character
line length.  The following code formatting and linting tools are run on all the
Python files (excluding the directories `ethosu/vela/tflite/` and
`ethosu/vela/ethos_u55_regs` because they contain auto-generated code):

* reorder-python-import (code formatter)
* black (code formatter)
* flake8 (code linter)

These tools are run using the [pre-commit](https://pre-commit.com/) framework.
This is also used to run the following test and coverage tools:

* pytest (testing framework)
* pytest-cov (code coverage plugin for pytest)

### Installation

To install pre-commit, pytest and pytest-cov in the pipenv virtual environment
use the following command:

```bash
pipenv install -e . --dev
```

The remaining tools will all be installed automatically upon first use.

### Add pre-commit hook (Automatically running the tools)

To support code development all the above tools can be configured to run
automatically on `git commit` (except pytest-cov which is run on `git push`) by
using the command:

```bash
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

When committing (or pushing) if any of the tools result in a failure (meaning an
issue was found) then it will need to be resolved and the git operation
repeated.

### Manually running the tools

All of the tools can be run individually by invoking them using the following
pre-commit framework commands:

```bash
$ pre-commit run reorder-python-imports --all-files
...
$ pre-commit run black --all-files
...
$ pre-commit run flake8 --all-files
...
$ pre-commit run pytest
...
$ pre-commit run pytest-cov --hook-stage push
...
```

Alternatively, all of the commit stage hooks can be run using the command:

```bash
$ pre-commit run --all-files
Reorder python imports...................................................Passed
black....................................................................Passed
flake8...................................................................Passed
pytest...................................................................Passed
```
