using Pkg
#Pkg.add("BSON")
#Pkg.add(PackageSpec(url=""))
#using BSON
using CSV, DataFrames, PyCall, Conda, LinearAlgebra, Statistics
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
jl = pyimport("joblib")

# inputing 28302 x 4 df
# columns: SMILES, INCHIKEY, CNLmasses, PRECURSOR_ION
inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES])