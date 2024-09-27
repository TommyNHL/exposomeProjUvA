## INPUT(S)
# dataAP2DFingerprinter.csv
# dataPubchemFingerprinter_converted.csv

## OUTPUT(S)
# dataAllFingerprinter_4RiPredict.csv

## install packages needed ##
using Pkg
#Pkg.add("PyCall")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using CSV, DataFrames, PyCall, Conda, LinearAlgebra, Statistics
Conda.add("pubchempy")
Conda.add("padelpy")
Conda.add("joblib")

## import packages from Python ##
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
jl = pyimport("joblib")

## import csv1 file- AP2D ##
# original input csv1 has 31402 rows (column header exclusive)
# 717 compound entries have no SMILES id -> conversion failed
# 1 compound entry has an invalid SMILES id -> conversion failed
# so the updated csv1 input is a 30684 x 782 df, columns include 
        #SMILES, INCHIKEY, 780 APC2D FPs
input_ap2d = CSV.read("F:\\dataAP2DFingerprinter.csv", DataFrame)

## import csv2 file- PubChem ##
# input csv2 is a 30684 x 160 df, columns include 
        # SMILES, INCHIKEY, 148 Pubchem FPs, and 10 newly converted columns
input_pub = CSV.read("F:\\dataPubchemFingerprinter_converted.csv", 
    DataFrame, select = Symbol.("PubchemFP".*string.(151:160)))

## replace missing values ##
ind_ap2d = vec(any(ismissing.(Matrix(input_ap2d)) .== 1, dims = 2))
ind_pub = vec(any(ismissing.(Matrix(input_pub)) .== 1, dims = 2))
ind_all = ind_ap2d .+ ind_pub
input_ap2d = input_ap2d[ind_all .== 0, :]
input_pub = input_pub[ind_all .== 0, :]

# merge/join 2 dfs
# output csv is a 30684 x 790 df, columns include 
        # SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs
allFPs = hcat(input_ap2d, input_pub)

# save the output table as a spreadsheet ##
savePath = "F:\\UvA\\dataAllFingerprinter_4RiPredict.csv"
CSV.write(savePath, allFPs)
