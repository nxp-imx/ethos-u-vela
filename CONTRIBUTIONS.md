<!--
SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>

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
# Vela Contributions

Contributions to Vela are very much welcomed!

## Coding Standard

Vela is written using Python 3.7 language constructs in order to aid
compatibility with other tools.  All code must also be run through the
formatting and linting tools described in [Vela Testing](TESTING.md)

## Submitting

In order to submit a contribution push your patch to the
[Vela Gerrit Server](https://review.mlplatform.org/q/project:ml%252Fethos-u%252Fethos-u-vela)
using the address `ssh://<GITHUB_USER_ID>@review.mlplatform.org:29418/ml/ethos-u/ethos-u-vela`.
To do this you will need to sign-in to the platform using a GitHub account and
add your SSH key under your settings. If there is a problem adding the SSH key make sure
there is a valid email address in the Email Addresses field.  
In the commit message please include a Change-Id and a Signed-off-by (described
below).


## Contribution Guidelines

Contributions are only accepted under the following conditions:

* You certify that the origin of the submission conforms to the
[Developer Certificate of Origin (DCO) V1.1](https://developercertificate.org/)
* You give permission according to the [Apache License 2.0](LICENSE.txt).

To indicate that you agree to these contribution guidelines you must add an
appropriate 'Signed-off-by: Real Name username@example.org' line with your real
name and e-mail address to every commit message.  This can be done automatically
by adding the `-s` option to your `git commit` command.

No contributions will be accepted from pseudonyms or anonymous sources.

## Code Reviews

All contributions go through a code review process.  Only submissions that are
approved and verified by this process will be accepted.  Code reviews are
performed using the
[Vela Gerrit Server](https://review.mlplatform.org/q/project:ml%252Fethos-u%252Fethos-u-vela).

## Testing Prior to Submission

Prior to submitting a patch for review please make sure that all the pre-commit
checks and tests have been run and are passing (see [Vela Testing](TESTING.md)
for more details).

## Bug Resolution

In the case that your submission aims to resolve a bug, please follow the
[Vela Community Bug Reporting Process](BUGS.md). The document guides
you through the process of adding a bug ticket to the bug tracker on the
[Maniphest website](https://developer.mlplatform.org/maniphest/).

The ticket will be visible to the public and will thus help the Vela community
track and find solutions to any bugs that are found during the use of Vela.

Please include a link to your patch in the bug ticket description.