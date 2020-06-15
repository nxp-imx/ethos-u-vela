# Vela Releases

These are the release notes for all Vela releases.  They document all of the
main feature changes, interface changes and reported defects that have been
fixed.  The version numbering adheres to the
[semantic versioning](https://semver.org/) scheme.

## Release 1.0 - 18/06/2020

**Main feature changes:**

* Int16 support
* New HW operator support: RESIZE_BILINEAR and TRANSPOSE_CONV
* NHCWB16 support between cascaded passes
* Stride 3 support
* pre-commit framework for code formatting, linting and testing

**Interface changes:**

* Removal of CLI options: `--batch-size`, `--inter-pass-cycle-delay`,
`--dram-bandwidth`
* Addition of CLI options: `--nhcwb16-between-cascaded-passes`

**Reported defect fixes:**

* MLCE-218: Crop2D operator not passing through optimizer
* MLCE-219: Custom Operator not passing through optimizer

## Release 0.1 - 29/04/2020

Initial release.
