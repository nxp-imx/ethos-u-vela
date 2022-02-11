# Vela Releases

These are the release notes for all Vela releases.  They document all of the
main feature changes, interface changes and reported defects that have been
fixed.  The version numbering adheres to the
[semantic versioning](https://semver.org/) scheme.

## Release 3.3.0 - 25/02/2022

**Main feature changes:**

* Upgrade TensorFlow Lite support to version 2.7
  * Increases Python requirement to at least version 3.8
* Scheduling algorithm improvements
  * Improved spilling reduces DRAM bandwidth for Ethos-U65 Dedicated SRAM
  * Improved weight buffering increases performance for Ethos-U65 512-MAC
  * Optimised tensor allocation reduces compilation time for large networks
* Extended operator support:
  * RESIZE_BILINEAR: Adds support for upscaling on NPU by factors of 4x4 and 8x8

**Interface changes:**

* None

**Reported defect fixes:**

* Memory corruption with custom operator and zero concatenation (MLCE-678)
* Crash when creating explicit padding (MLCE-684)
* Fix assert when setting address on identical LUT tensors (MLCE-691)
* Reduce SRAM usage for some elementwise operations (MLCE-750)
* Issue when running a model with Padding (MLCE-768)

## Release 3.2.0 - 26/11/2021

**Main feature changes:**

* Bug fixes
* Documentation of community bug reporting via Phabricator on ML Platform
* New operator support: EXPAND_DIMS
* Extended experimental TOSA support

**Interface changes:**

* External API v1.2
  * Fixed bug in `npu_find_block_configs()` with Conv1D optimisation

* Verbose output
  * Clarified Subgraph IO Summary

* Summary output
  * Removed incorrect passes information
  * Fixed and reformatted the reporting of CPU operators

**Reported defect fixes:**

* Clarified section on access cycles in PERFORMANCE.md (MLCE-654)
* Crash due to mismatch in padding shape in OFM (MLCE-652)
* Bug where back to back activations were ignored (MLCE-650)
* Output mismatch for unsupported RESHAPE with 16-bit 5D tensors (MLCE-630)
* Bug where tensors with "skip connections" were incorrectly handled (MLCE-621)
* Crash due to the same tensor being used for both inputs to operator (MLCE-620)
* Missing constraint on batch size for MEAN operator (MLCE-619)
* Bug where third-party custom ops were not passed through correctly (MLCE-602)
* Crash due to mismatch in tensor indices for EXP operator (MLCE-599)

## Release 3.1.0 - 30/08/2021

**Main feature changes:**

* New operator support: SQUEEZE
* Updated support for TensorFlow 2.5
* Experimental limited support for reading TOSA files

**Interface changes:**

* Re-addition of CLI option:
  * `--recursion-limit`
* External API v1.1
  * Re-instated functionality of `find_block_configs()` from Vela 2.1.0

**Reported defect fixes:**

* Bug with IFM box depth for convolutions fused with SPLIT (MLCE-490)
* Bug with missing attribute from integer type (MLCE-534)
* Bug with incorrect options in TensorFlow Lite mapping (MLCE-427)

## Release 3.0.0 - 28/05/2021

**Main feature changes:**

* Improved user control over Vela
  * Allowing user to specify an arena cache size target
  * Allowing user to optimise for inference time or memory footprint
* Extended operator support for PAD and MEAN
* Multiple improvements to reduce compilation time

**Interface changes:**

* Addition of CLI options:
  * `--optimise`, `--arena-cache-size`
* Removal of CLI options:
  * `--cascading`, `--cache-bias-scale-tensor`, `--ifm-streaming`,
  * `--force-block-config`, `--block-config-limit`, `--recursion-limit`
  * `--nhcwb16-between-cascaded-passes`, `--weight-estimation-scaling`
  * `--pareto-metric`, `--verbose-pareto-frontier-schedules`

**Reported defect fixes:**

* Regression in FullyConnected between v2.0.1 and v2.1.0 (MLCE-484)
* Output mismatch for 16-bit TANH and SIGMOID (MLCE-362)
* Improved `--verbose-graph` CLI option output (MLCE-482)
* Bug with `--verbose-operators` CLI option (MLCE-444)
* Bug with incorrect tensor format chosen for SPLIT (MLCE-331)
* Bug with STRIDED_SLICE padding (MLCE-425)
* Bug with RESHAPE at the edge of Ethos-U custom operator (MLCE-443)
* Document Vela memory configuration and options (MLCE-410 & MLCE-498)

## Release 2.1.0 - 25/02/2021

**Main feature changes:**

* New operator support: PAD, MEAN and HARD_SWISH
* New HillClimb tensor allocator (default)
* Network performance improved of shallow layers
* Updated support to TensorFlow Lite 2.4
* Added Microsoft Windows 10 support
* Extended debug database to support multiple Ethos-U Custom operators
* Added cascading support to RESIZE_BILINEAR
* Improved performance estimations

**Interface changes:**

* Addition of Vela configuration (.ini) file options:
  * `MEM_burst_length`, `MEM_read_latency`, `MEM_write_latency`
* Change to CLI options:
  * `--tensor-allocator` (change default to HillClimb)
* Addition of CLI options:
  * `--verbose-all`

**Reported defect fixes:**

* Bug with handling multiple custom operators (MLCE-329)
* Bug with configuring Ethos-U55 with DRAM (MLCE-322)

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
  * Command Stream generation, Driver Payload creation, and Finding a Block
Config

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
