# ColabMPNN

[![Build Status](https://github.com/MurrellGroup/ColabMPNN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/ColabMPNN.jl/actions/workflows/CI.yml?query=branch%3Amain)

ColabMPNN is a Julia wrapper for the MPNN submodule of the ColabDesign Python package, which can be found at https://github.com/sokrypton/ColabDesign/tree/main. 

For more details about usage and function arguments, see the [original Python documentation](https://github.com/sokrypton/ColabDesign/blob/main/mpnn/README.md)

## Installation

Add ColabMPNN to your Julia environment in the REPL:
```
]add https://github.com/MurrellGroup/ColabMPNN.jl
```

## Usage

Create a model using the `mk_mpnn_model` function.

```julia
mpnn_model = mk_mpnn_model()
```

In order to sample, chains from a PDB file must first be prepared.

```julia
prep_inputs(mpnn_model, pdb_filename="example.pdb", chain="A")
```

Sample sequences using the `sample` function, or in parallel with `sample_parallel`, with the model as the first argument. These functions return a `Samples` instance.

```julia
samples = sample_parallel(mpnn_model, batch=10, temperature=0.1)
```

Sampling returns a `Samples` instance with the following fields:
- `seq::Vector{String}`
- `seqid::Vector{Float64}`
- `score::Vector{Float64}`
- `logits::Array{Float32, 3}`
- `decoding_order::Array{Int32, 3}`
- `S::Array{Float32, 3}`