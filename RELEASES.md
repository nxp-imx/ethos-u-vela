# Vela Releases

These are the release notes for all Vela releases.  They document all of the
main feature changes, interface changes and reported defects that have been
fixed.  The version numbering adheres to the
[semantic versioning](https://semver.org/) scheme.

## Release 2.0.1 - 03/12/2020

* Fixed a broken link in the long description used by PyPi

## Release 2.0.0 - 30/11/2020

**Main feature changes:**

* Advanced Spilling for Ethos-U65 Dedicated SRAM
* Specific network performance improvements for Ethos-U65 Dedicated SRAM
* General performance improvements for different networks and system
configurations
* Support SOFTMAX batching
* Improved supported operator checking and reporting
* Improved pass-through of Ethos-U hardware unsupported operators
* Enhanced performance estimations
* Extended external API
* New Vela configuration file format (with example `vela.ini`)
* Updated support to TensorFlow Lite 2.3
* Made 8-bit TANH and LOGISTIC bit-exact and faster
* Generation of a network debug database to aid profiling and debug

**Interface changes:**

* Non-backwards compatible changes to the Vela configuration (.ini) file
* Removal of CLI options:
  * `--show-minimum-possible-allocation`, `--ifm-ofm-overlap`
  * `--permanent-storage`, `--global-memory-clock-scale`
* Change to CLI options:
  * `--config` (support multiple)
  * `--accelerator-config` (change 'yoda' to 'ethos-u65')
* Addition of CLI options:
  * `--api-version`, `--supported-ops-report`, `--enable-debug-db`
  * `--verbose-config`, `--cache-bias-scale-tensor`, `--memory-mode`
  * `--cpu-tensor-alignment`
* Addition of External APIs:
  * Command Stream generation, Driver Payload creation, and Finding a Block Config

**Reported defect fixes:**

* Add support for 128-Byte alignment (`--cpu-tensor-alignment`) (MLCE-221)
* Bug with SPLIT and non-unique scale & bias tensors (MLCE-234)
* Bug with overlapping tensor buffers in CONCATENATION (MLCE-246)
* Bug with a non-splitting SPLIT (MLCE-247)
* Regression in STRIDED_SLICE between v1.0.0 and v1.2.0 (MLCE-252)
* Add batch support to SOFTMAX (MLCE-265)
* Vela config file examples (MLCE-277)

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
