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

# input csv is a 30684 x 793 df, columns include 
        #SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs, 
        #and newly added one (FP-derived predicted Ri)
inputDBcocamide = CSV.read("D:\\0_data\\dataAllFP_withNewPredictedRi.csv", DataFrame)
sort!(inputDBcocamide, [:INCHIKEY, :SMILES])

# df creating
dfOnlyCocamides = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
dfWithoutCocamides = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
for col in names(inputDB)[3:end]
    dfOnlyCocamides[:, col] = []
    dfWithoutCocamides[:, col] = []
end
dfOnlyCocamides[:, "predictRi"] = []
dfWithoutCocamides[:, "predictRi"] = []
size(dfOnlyCocamides)  # 0 x (15979+1)
size(dfWithoutCocamides)  # 0 x (15979+1)

function cocamidesOrNot(i)
    if (inputDB[i, "INCHIKEY"] in Array(inputDBcocamide[:, "INCHIKEY"]))
        return true
    else
        if (inputDB[i, "SMILES"] in Array(inputDBcocamide[:, "SMILES"]))
            return true
        end
    end
end

function findRowNumber(i)
    if (inputDB[i, "INCHIKEY"] in Array(inputDBcocamide[:, "INCHIKEY"]))
        println(findall(inputDBcocamide.INCHIKEY .== inputDB[i, "INCHIKEY"]))
        return findall(inputDBcocamide.INCHIKEY .== inputDB[i, "INCHIKEY"])
    else
        if (inputDB[i, "SMILES"] in Array(inputDBcocamide[:, "SMILES"]))
            println(findall(inputDBcocamide.SMILES .== inputDB[i, "SMILES"]))
            return findall(inputDBcocamide.SMILES .== inputDB[i, "SMILES"])
        end
    end
end
    
function dfExtract(i, columnsCNLs)
    temp = []
    push!(temp, inputDB[i, "SMILES"])
    push!(temp, inputDB[i, "INCHIKEY"])
    for col in columnsCNLs
        push!(temp, inputDB[i, col])
    end
    return temp
end

for i in 1:size(inputDB, 1)
    if (cocamidesOrNot(i) == true)
        tempRow = dfExtract(i, names(inputDB)[3:end])
        push!(tempRow, inputDBcocamide[findRowNumber(i)[end:end], "predictRi"][1])
        push!(dfOnlyCocamides, tempRow)
    else
        tempRow = dfExtract(i, names(inputDB)[3:end])
        push!(tempRow, Float64(0))
        push!(dfWithoutCocamides, tempRow)
    end
end

# df with 30684 x 2+15977+1
dfOnlyCocamides
# ouputing df 28302 x (2+15977)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamides.csv"
CSV.write(savePath, dfOnlyCocamides)

# df with 28302 or less x 2+15977+1
dfWithoutCocamides
# ouputing df 28302 x (2+15977)
savePath = "D:\\0_data\\dataframeCNLsRows_dfWithoutCocamides.csv"
CSV.write(savePath, dfWithoutCocamides)