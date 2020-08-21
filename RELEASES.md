# Vela Releases

These are the release notes for all Vela releases.  They document all of the
main feature changes, interface changes and reported defects that have been
fixed.  The version numbering adheres to the
[semantic versioning](https://semver.org/) scheme.

## Release 1.2.0 - 31/08/2020

**Main feature changes:**

* New Ethos-U NPU operator support: SOFTMAX and QUANTIZE (requantise only)
* Improved Ethos-U NPU operator support: RESIZE_BILINEAR and LEAKY_RELU
* Improved pass-through of operators not supported by Ethos-U NPU
* Preserve TensorFlow Lite (.tflite) file metadata
* Improved network performance
* Performance estimation statistics improved
* New external API

**Interface changes:**

* Addition of CLI options: `--weight-estimation-scaling`
* Addition of External APIs: Weight compression and Bias & scale encoding

**Reported defect fixes:**

* Custom operator not passing through Vela (MLCE-223)
* Flash usage increases in 1.1.0 (MLCE-227)
* Vela fails with optional bias tensor (MLCE-231)

## Release 1.1.0 - 30/06/2020

**Main feature changes:**

* Multi-core support
* DRAM memory allocation support

**Interface changes:**

* Change to CLI options: `--accelerator-config` (added values `yoda-256` and
`yoda-512`)

## Release 1.0.0 - 18/06/2020

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

* Crop2D operator not passing through optimizer (MLCE-218)
* Custom Operator not passing through optimizer (MLCE-219)

## Release 0.1.0 - 29/04/2020

Initial release.
