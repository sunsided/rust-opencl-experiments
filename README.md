# OpenCL Experiments (WIP)

Experiments for running streaming dot products on GPU using
OpenCL in Rust.

## Vector Database

A binary vector database format is added to provide basic testing data.
The format of `vectors.bin` is

| Length | Content              | Example Value |
|--------|----------------------|---------------|
| 4      | Version              | 0             |
| 4      | `u32::MAX`           | unused        |
| 4      | Number of vectors    | 1000000       |
| 4      | Number of dimensions | 4096          |

The [bins/fetch_vectors](bins/fetch_vectors/src/main.rs) script is one
implementation for fetching data from a proprietary data source.
