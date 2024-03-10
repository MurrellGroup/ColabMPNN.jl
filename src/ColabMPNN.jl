module ColabMPNN

export mpnn_sample

import Pkg
using PyCall, Conda

const colabdesign = PyNULL()
const mpnn_model = PyNULL()

function __init__()
    ENV["PYTHON"] = ""
    Pkg.build("PyCall")

    if !haskey(Conda._installed_packages_dict(), "colabdesign")
        Conda.pip_interop(true)
        Conda.pip("install", "git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
        Conda.add("colabdesign")
    end

    copy!(colabdesign, pyimport_conda("colabdesign", "colabdesign"))
    copy!(mpnn_model, colabdesign.mk_mpnn_model())
end

function mpnn_sample(path::String, temp::Float64, chain="A")
    mpnn_model.prep_inputs(pdb_filename=path, chain=chain)
    samples = mpnn_model.sample_parallel(temperature=temp)
    return String.(collect(samples["seq"]))
end

end
