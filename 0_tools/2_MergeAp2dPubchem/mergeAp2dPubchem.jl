using Pkg
Pkg.add("BSON")
Pkg.add(PackageSpec(url=""))
using BSON
using CSV, DataFrames, PyCall, Conda, LinearAlgebra, Statistics
Conda.add("joblib")
## import packages ##
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
jl = pyimport("joblib")

# import files
input_ap2d = CSV.read("D:\\0_data\\dataAP2DFingerprinter.csv", DataFrame)
input_pub = CSV.read("D:\\0_data\\dataPubchemFingerprinter_converted.csv", 
    DataFrame, select = Symbol.("PubchemFP".*string.(151:160)))

# replacing missing values
ind_ap2d = vec(any(ismissing.(Matrix(input_ap2d)) .== 1, dims = 2))
ind_pub = vec(any(ismissing.(Matrix(input_pub)) .== 1, dims = 2))
ind_all = ind_ap2d .+ ind_pub

input_ap2d = input_ap2d[ind_all .== 0, :]
input_pub = input_pub[ind_all .== 0, :]

# merge
allFPs = hcat(input_ap2d, input_pub)

# save
savePath = "D:\\0_data\\dataAllFingerprinter_4RiPredict.csv"
CSV.write(savePath, allFPs)