using Pkg
Pkg.add("BSON")
#Pkg.add(PackageSpec(url=""))
using BSON
using CSV, DataFrames, PyCall, Conda, LinearAlgebra, Statistics
Conda.add("pubchempy")
Conda.add("padelpy")
Conda.add("joblib")
## import packages ##
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
jl = pyimport("joblib")

function convertPubChemFPs(ACfp::DataFrame, PCfp::DataFrame)
    FP1tr = ACfp
    pubinfo = Matrix(PCfp)

    #ring counts
    FP1tr[!,"PCFP-r3"] = pubinfo[:,1]
    FP1tr[!,"PCFP-r3"][pubinfo[:,8] .== 1] .= 2
    FP1tr[!,"PCFP-r4"] = pubinfo[:,15]
    FP1tr[!,"PCFP-r4"][pubinfo[:,22] .== 1] .= 2
    FP1tr[!,"PCFP-r5"] = pubinfo[:,29]
    FP1tr[!,"PCFP-r5"][pubinfo[:,36] .== 1] .= 2
    FP1tr[!,"PCFP-r5"][pubinfo[:,43] .== 1] .= 3
    FP1tr[!,"PCFP-r5"][pubinfo[:,50] .== 1] .= 4
    FP1tr[!,"PCFP-r5"][pubinfo[:,57] .== 1] .= 5

    FP1tr[!,"PCFP-r6"] = pubinfo[:,64]
    FP1tr[!,"PCFP-r6"][pubinfo[:,71] .== 1] .= 2
    FP1tr[!,"PCFP-r6"][pubinfo[:,78] .== 1] .= 3
    FP1tr[!,"PCFP-r6"][pubinfo[:,85] .== 1] .= 4
    FP1tr[!,"PCFP-r6"][pubinfo[:,92] .== 1] .= 5
    FP1tr[!,"PCFP-r7"] = pubinfo[:,99]
    FP1tr[!,"PCFP-r7"][pubinfo[:,106] .== 1] .= 2
    FP1tr[!,"PCFP-r8"] = pubinfo[:,113]
    FP1tr[!,"PCFP-r8"][pubinfo[:,120] .== 1] .= 2
    FP1tr[!,"PCFP-r9"] = pubinfo[:,127]
    FP1tr[!,"PCFP-r10"] = pubinfo[:,134]

    #minimum number of type of rings
    arom = zeros(size(pubinfo,1))
    arom[(arom .== 0) .& (pubinfo[:,147] .== 1)] .= 4
    arom[(arom .== 0) .& (pubinfo[:,145] .== 1)] .= 3
    arom[(arom .== 0) .& (pubinfo[:,143] .== 1)] .= 2
    arom[(arom .== 0) .& (pubinfo[:,141] .== 1)] .= 1
    FP1tr[!,"minAromCount"] = arom
    het = zeros(size(pubinfo,1))
    het[(het .== 0) .& (pubinfo[:,148] .== 1)] .= 4
    het[(het .== 0) .& (pubinfo[:,146] .== 1)] .= 3
    het[(het .== 0) .& (pubinfo[:,144] .== 1)] .= 2
    het[(het .== 0) .& (pubinfo[:,142] .== 1)] .= 1
    FP1tr[!,"minHetrCount"] = het

    return FP1tr
end


#load all data
# original input csv has 31402 rows (column header exclusive)
# 717 compound entries have no SMILES id -> conversion failed
# 1 compound entry has an invalid SMILES id -> conversion failed
# so the updated csv input is a 30684 x 150 df, columns include 
        #SMILES, INCHIKEY, and 148 Pubchem FPs, 148 -> 10 columns after operation
input = CSV.read("F:\\dataPubchemFingerprinter.csv", DataFrame)
output = convertPubChemFPs(input[:,1:end], input[:,3:end])
# output csv is a 30684 x 160 df, columns include 
        #SMILES, INCHIKEY, 148 Pubchem FPs, and 10 newly added columns

savePath = "F:\\UvA\\dataPubchemFingerprinter_converted.csv"
CSV.write(savePath, output)