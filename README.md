# ColabMPNN

[![Build Status](https://github.com/MurrellGroup/ColabMPNN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/ColabMPNN.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia wrapper for ColabDesign's MPNN module using PyCall.

For more details, see the [original Python documentation](https://github.com/sokrypton/ColabDesign/blob/main/mpnn/README.md)

## Installation

Add ColabMPNN to your Julia environment in the REPL:
```
]add https://github.com/MurrellGroup/ColabMPNN.jl
```

## Usage

Create a model using the `mk_mpnn_model` function. See arguments in the [Python code](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/mpnn/model.py#L24).

```julia
mpnn_model = mk_mpnn_model()
```

Inputs are prepared to a model using, with the model as first argument. See

```julia
prep_inputs(mpnn_model, pdb_filename="example.pdb", chain="A")
```

Sample sequences using the `sample` function, or in parallel with `sample_parallel`, with the model as the first argument. These functions return a `Samples` instance.

```julia
samples = sample_parallel(mpnn_model)
```

The `Samples` type has the following fields:
- `seq::Vector{String}`
- `seqid::Vector{Float64}`
- `score::Vector{Float64}`
- `logits::Array{Float32, 3}`
- `decoding_order::Array{Int32, 3}`
- `S::Array{Float32, 3}`