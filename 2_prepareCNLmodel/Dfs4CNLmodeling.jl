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

# other input 
inputAllMS2DB = CSV.read("D:\\0_data\\databaseOfAllMS2_withMergedNLs.csv", DataFrame)
sort!(inputAllMS2DB, [:INCHIKEY, :SMILES])

inputAllFPDB = CSV.read("D:\\0_data\\dataAllFP_withNewPredictedRi.csv", DataFrame)
sort!(inputAllFPDB, [:INCHIKEY, :SMILES])

inputCocamidesTrain = CSV.read("D:\\0_data\\CocamideExt_Fingerprints_train.csv", DataFrame)
sort!(inputCocamidesTrain, :SMILES)

inputCocamidesTest = CSV.read("D:\\0_data\\CocamideExt_Fingerprints_test.csv", DataFrame)
sort!(inputCocamidesTest, :SMILES)


# df creating
dfOnlyCocamides = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
dfOutsideCocamides = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
for col in names(inputDB)[3:end]
    dfOnlyCocamides[:, col] = []
    dfOutsideCocamides[:, col] = []
end
dfOnlyCocamides[:, "predictRi"] = []
dfOutsideCocamides[:, "predictRi"] = []
size(dfOnlyCocamides)  # 0 x (15979+1)
size(dfOutsideCocamides)  # 0 x (15979+1)

function cocamidesOrNot(DB, i)
    if (DB[i, "SMILES"] in Array(inputCocamidesTrain[:, "SMILES"]) || DB[i, "SMILES"] in Array(inputCocamidesTest[:, "SMILES"]))
        return true
    else
        return false
    end
end

function haveFPRiOrNot(DB, i)
    if (DB[i, "INCHIKEY"] in Array(inputAllFPDB[:, "INCHIKEY"]) || DB[i, "SMILES"] in Array(inputAllFPDB[:, "SMILES"]))
        return true
    else 
        return false
    end
end

function findRowNumber4Ri(DB, i)
    if (DB[i, "INCHIKEY"] in Array(inputAllFPDB[:, "INCHIKEY"]))
        return findall(inputAllFPDB.INCHIKEY .== DB[i, "INCHIKEY"])
    elseif (DB[i, "SMILES"] in Array(inputAllFPDB[:, "SMILES"]))
        return findall(inputAllFPDB.SMILES .== DB[i, "SMILES"])
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

inputDB
for i in 1:size(inputDB, 1)
    if (cocamidesOrNot(inputDB, i) == true)  # && haveFPRiOrNot(inputDB, i) == true)
        tempRow = dfExtract(i, names(inputDB)[3:end])
        push!(tempRow, inputAllFPDB[findRowNumber(i)[end:end], "predictRi"][1])
        push!(dfOnlyCocamides, tempRow)
    elseif (cocamidesOrNot(inputDB, i) == false)  # && haveFPRiOrNot(inputDB, i) == true)
        tempRow = dfExtract(i, names(inputDB)[3:end])
        push!(tempRow, Float64(0))
        push!(dfOutsideCocamides, tempRow)
    end
end
dfOnlyCocamides
dfOutsideCocamides

inputAllMS2DB
countOnlyCocamides = 0
countOutsideCocamides = 0
for i in 1:size(inputAllMS2DB, 1)
    if (cocamidesOrNot(inputAllMS2DB, i) == true)  # && haveFPRiOrNot(inputAllMS2DB, i) == true)
        countOnlyCocamides += 1
    elseif (cocamidesOrNot(inputAllMS2DB, i) == false)  # && haveFPRiOrNot(inputAllMS2DB, i) == true)
        countOutsideCocamides +=1
    end
end
countOnlyCocamides
countOutsideCocamides

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