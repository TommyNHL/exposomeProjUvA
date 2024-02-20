using Pkg
#Pkg.add("BSON")
#Pkg.add(PackageSpec(url=""))
#using BSON
using CSV, DataFrames #, PyCall, Conda, LinearAlgebra, Statistics
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#pcp = pyimport("pubchempy")
#pd = pyimport("padelpy")
#jl = pyimport("joblib")

# inputing 693677 x 4 df
# columns: SMILES, INCHIKEY, PRECURSOR_ION, CNLmasses...
inputDB = CSV.read("D:\\0_data\\databaseOfInternal_withNLs.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES, :PRECURSOR_ION, :CNLmasses])

# creating a table with 3 columns
dfOutput = DataFrame([[],[],[]], ["SMILES", "INCHIKEY", "FREQUENCY"])
size(dfOutput)

count = 0
str = string(inputDB[1, "SMILES"], inputDB[1, "INCHIKEY"])
for i in 1:size(inputDB, 1)
    if (string(inputDB[i, "SMILES"], inputDB[i, "INCHIKEY"]) == str)
        count += 1
    elseif (i == size(inputDB, 1))
        temp = []
        push!(temp, inputDB[i, "SMILES"])
        push!(temp, inputDB[i, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput, temp)
    else
        temp = []
        push!(temp, inputDB[i, "SMILES"])
        push!(temp, inputDB[i, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput, temp)
        str = string(inputDB[i, "SMILES"], inputDB[i, "INCHIKEY"])
        count = 1
    end
end

# 29290 x 3
dfOutput
# save
# output csv is a 693677 x 4 df
savePath = "D:\\0_data\\countingRows4Leverage.csv"
CSV.write(savePath, dfOutput)