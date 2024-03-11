module ColabMPNN

export mpnn, mk_mpnn_model, prep_inputs, sample, sample_parallel, score, get_unconditional_logits

import Pkg
using Conda, PyCall

const mpnn = PyNULL()

function __init__()
    ENV["PYTHON"] = ""
    Pkg.build("PyCall")

    if !haskey(Conda._installed_packages_dict(), "colabdesign")
        Conda.pip_interop(true)
        Conda.pip("install", "git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
        Conda.add("colabdesign")
    end

    copy!(mpnn, pyimport_conda("colabdesign.mpnn", "colabdesign"))
end

mk_mpnn_model(; kwargs...) = mpnn.mk_mpnn_model(; kwargs...)

prep_inputs(mpnn_model; kwargs...) = mpnn_model.prep_inputs(; kwargs...)

sample(mpnn_model; kwargs...) = mpnn_model.sample(; kwargs...)

sample_parallel(mpnn_model; kwargs...) = mpnn_model.sample_parallel(; kwargs...)

score(mpnn_model; kwargs...) = mpnn_model.score(; kwargs...)

get_unconditional_logits(mpnn_model) = mpnn_model.get_unconditional_logits()

end
