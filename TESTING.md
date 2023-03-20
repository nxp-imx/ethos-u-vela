<!--
SPDX-FileCopyrightText: Copyright 2020, 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the License); you may
not use this file except in compliance with the License.
You may obtain a copy of the License at

www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Vela Testing

## Tools

Vela's Python codebase is PEP8 compliant with the exception of a 120 character
line length.  The following code formatting and linting tools are run on all the
Python files (excluding the directories `ethosu/vela/tflite/`, `ethosu/vela/tosa/`,
and `ethosu/vela/ethos_u55_regs` because they contain auto-generated code):

* mypy (code linter)
* reorder-python-import (code formatter)
* black (code formatter)
* flake8 (code linter)
* pylint (code linter)

These tools are run using the [pre-commit](https://pre-commit.com/) framework.
This is also used to run the following test and coverage tools:

* pytest (testing framework)
* pytest-cov (code coverage plugin for pytest)

### Installation

To install the development dependencies, use the following command:

``` bash
pip install -e .[dev]
```

This command will install the following tools:

* pytest
* pytest-cov
* pre-commit
* build
* setuptools_scm

The remaining tools will all be installed automatically upon first use of pre-commit.

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
$ pre-commit run mypy --all-files
...
$ pre-commit run reorder-python-imports --all-files
...
$ pre-commit run black --all-files
...
$ pre-commit run flake8 --all-files
...
$ pre-commit run pylint --all-files
...
$ pre-commit run pytest
...
$ pre-commit run pytest-cov --hook-stage push
...
```

Alternatively, all of the commit stage hooks can be run using the command:

```bash
$ pre-commit run --all-files
mypy.....................................................................Passed
Reorder python imports...................................................Passed
black....................................................................Passed
flake8...................................................................Passed
pylint...................................................................Passed
pytest...................................................................Passed
```
