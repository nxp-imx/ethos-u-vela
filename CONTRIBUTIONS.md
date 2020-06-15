# Vela Contributions

Contributions to Vela are very much welcomed!

## Coding Standard

Vela requires Python 3.6 to run. However, it is written using only version 3.5
language constructs with the addition of Enums (hence requiring 3.6 to run).
The aim is to maintain this in order to aid compatibility with other tools.

## Submitting

In order to submit a contribution please submit a Change Request (CR) to the
[Vela Gerrit Server](https://review.mlplatform.org/q/project:ml%252Fethos-u%252Fethos-u-vela).
To do this you will need to sign-in to the platform using a GitHub account.

## Contribution Guidelines

Contributions are only accepted under the following conditions:

* You certify that the origin of the submission conforms to the
[Developer Certificate of Origin (DCO) V1.1](https://developercertificate.org/)
* You give permission according to the [Apache License 2.0](LICENSE.txt).

To indicate that you agree to the contribution guidelines you must add an
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
