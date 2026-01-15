module ColabMPNN

export mpnn
export Samples, Score
export mk_mpnn_model, prep_inputs, sample, sample_parallel, score, get_unconditional_logits

using PythonCall

const mpnn = PythonCall.pynew()

__init__() = PythonCall.pycopy!(mpnn, pyimport("colabdesign.mpnn"))

struct Samples
    seq::Vector{String}
    seqid::Vector{Float64}
    score::Vector{Float64}
    logits::Array{Float32, 3}
    decoding_order::Array{Int32, 3}
    S::Array{Float32, 3}

    function Samples(samples::PyDict)
        new([pyconvert(Any, samples[Py(string(f))]) for f in fieldnames(Samples)]...)
    end
end

struct Score
    seqid::Float64
    score::Float64
    logits::Array{Float32, 2}
    decoding_order::Array{Int32, 1}
    S::Array{Float32, 2}

    function Score(scores::PyDict)
        new([pyconvert(Any, scores[pystr(string(f))]) for f in fieldnames(Score)]...)
    end
end

# Due to how function arguments differ between Python and Julia, the arguments can be specified as either positional or keyword arguments.

"""
    mk_mpnn_model(
        model_name="v_48_020",
        backbone_noise=0.0,
        dropout=0.0,
        seed=nothing,
        verbose=false,
        weights="original",
    )
"""
mk_mpnn_model(args...; kwargs...) = mpnn.mk_mpnn_model(args...; kwargs...)

"""
    prep_inputs!(mpnn_model,
        pdb_filename=nothing,
        chain=nothing,
        homooligomer=false,
        ignore_missing=true,
        fix_pos=nothing,
        inverse=false,
        rm_aa=nothing,
        verbose=false,
    )
"""
prep_inputs!(mpnn_model, args...; kwargs...) = mpnn_model.prep_inputs(args...; kwargs...)

"""
    sample(mpnn_model,
        num=1,
        batch=1,
        temperature=0.1,
        rescore=false,
    )
"""
sample(mpnn_model, args...; kwargs...) = Samples(PyDict(mpnn_model.sample(args...; kwargs...)))

"""
    sample_parallel(mpnn_model,
        batch=10,
        temperature=0.1,
        rescore=false,
    )
"""
sample_parallel(mpnn_model, args...; kwargs...) = Samples(PyDict(mpnn_model.sample_parallel(args...; kwargs...)))

"""
    score(mpnn_model,
        seq=nothing,
    )
"""
score(mpnn_model, args...; kwargs...) = Score(PyDict(mpnn_model.score(args...; kwargs...)))

"""
    get_unconditional_logits(mpnn_model)
"""
get_unconditional_logits(mpnn_model) = pyconvert(Array, mpnn_model.get_unconditional_logits())

end
