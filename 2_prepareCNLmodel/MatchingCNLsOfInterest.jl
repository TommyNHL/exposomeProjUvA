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

# filtering in CNLs features according to the pre-defined CNLs in CNLs_10mDa.csv
# inputing 16022 candidates
candidatesList = Array(CSV.read("D:\\0_data\\CNLs_10mDa.csv", DataFrame)[:,1])

# inputing 2521 x 20496 df
# columns: SMILES, INCHIKEY, 20494 CNLs
inputDB = CSV.read("D:\\0_data\\dataframeScoresCNLsTPTNwithoutSelection.csv", DataFrame)

dfCandidates = deepcopy(inputDB)
size(dfCandidates)

for col in names(inputDB)[3:end]
    if (Float64[BigFloat(col)][1] in candidatesList)
        continue
    else
        select!(dfCandidates, Not(col))
    end
end
size(dfCandidates)
dfCandidates

# output csv is a 2521 x 11519 df, 11519 / 16022 CNLs features
savePath = "D:\\0_data\\dataframeScoresCNLsTPTNwithSelection.csv"
CSV.write(savePath, dfCandidates)