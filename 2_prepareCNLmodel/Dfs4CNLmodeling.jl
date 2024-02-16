using Pkg
#Pkg.add("BSON")
#Pkg.add(PackageSpec(url=""))
#using BSON
using CSV, DataFrames, PyCall, Conda, LinearAlgebra, Statistics
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
#Conda.add("Numpy")
## import packages ##
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
jl = pyimport("joblib")
np = pyimport("numpy")

# inputing 28302 x (3+15994) df
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows.csv", DataFrame)

# imputing 30684 x (2+791) df, columns include 
        #SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs, 
        #and newly added one (FP-derived predicted Ri)
inputAllFPDB = CSV.read("D:\\0_data\\dataAllFP_withNewPredictedRi.csv", DataFrame)
sort!(inputAllFPDB, [:INCHIKEY, :SMILES])

# inputing dfs for separation of the cocamides and non-cocamides datasets
## 5364 x 931 df 
inputCocamidesTrain = CSV.read("D:\\0_data\\CocamideExt_Fingerprints_train.csv", DataFrame)
sort!(inputCocamidesTrain, :SMILES)

## 947 x 931 df
inputCocamidesTest = CSV.read("D:\\0_data\\CocamideExt_Fingerprints_test.csv", DataFrame)
sort!(inputCocamidesTest, :SMILES)

# df creating
dfOnlyCocamides = DataFrame([[],[],[]], ["ENTRY", "SMILES", "INCHIKEY"])
dfOutsideCocamides = DataFrame([[],[],[]], ["ENTRY", "SMILES", "INCHIKEY"])
for col in names(inputDB)[4:end]
    dfOnlyCocamides[:, col] = []
    dfOutsideCocamides[:, col] = []
end
dfOnlyCocamides[:, "FPpredictRi"] = []
dfOutsideCocamides[:, "FPpredictRi"] = []
size(dfOnlyCocamides)  # 0 x (3+15994+1)
size(dfOutsideCocamides)  # 0 x (3+15994+1)

function cocamidesOrNot(DB, i)
    if (DB[i, "SMILES"] in Array(inputCocamidesTrain[:, "SMILES"]) || DB[i, "SMILES"] in Array(inputCocamidesTest[:, "SMILES"]))
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
    push!(temp, inputDB[i, "ENTRY"])
    push!(temp, inputDB[i, "SMILES"])
    push!(temp, inputDB[i, "INCHIKEY"])
    for col in columnsCNLs
        push!(temp, inputDB[i, col])
    end
    return temp
end

# x 15997
inputDB

for i in 1:size(inputDB, 1)
    if (cocamidesOrNot(inputDB, i) == true)
        tempRow = dfExtract(i, names(inputDB)[4:end])
        push!(tempRow, inputAllFPDB[findRowNumber4Ri(inputDB, i)[end:end], "predictRi"][1])
        push!(dfOnlyCocamides, tempRow)
    elseif (cocamidesOrNot(inputDB, i) == false)
        tempRow = dfExtract(i, names(inputDB)[4:end])
        push!(tempRow, inputAllFPDB[findRowNumber4Ri(inputDB, i)[end:end], "predictRi"][1])
        push!(dfOutsideCocamides, tempRow)
    end
end

# 28 x 15998 df
dfOnlyCocamides

# 4862 x 15998 df
dfOutsideCocamides

# outputing dfOnlyCocamides with 28 x (3+15994+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamides.csv"
CSV.write(savePath, dfOnlyCocamides)

# outputing dfOutsideCocamides with 4862 x (3+15994+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOutsideCocamides.csv"
CSV.write(savePath, dfOutsideCocamides)

# determining Leverage values
# 3+15994+1+1
dfOnlyCocamides[!, "LeverageValue"] .= Float64(0)
dfOutsideCocamides[!, "LeverageValue"] .= Float64(0)

###
function leverageCal(matX)
    hiis = []
    matX_t = transpose(matX)
    for i in 1:size(matX, 1)
        for j in 1:size(matX, 2)
            hii = (matX[i] ./ matX) * (matX[j] ./ matX_t)
            push!(hiis, hii)
        end
    end
    return hiis
end
###

levOnlyCocamides = leverageCal(Matrix(dfOnlyCocamides[:, 4:end-2]))
levOutsideCocamides = leverageCal(Matrix(dfOutsideCocamides[:, 4:end-2]))

diff = 0.001
bins = collect(0:diff:1)
counts = zeros(length(bins)-1)
for i = 1:length(bins)-1
    counts[i] = sum(bins[i] .<= levOnlyCocamides .< bins[i+1])
end
thresholdAD = bins[findfirst(cumsum(counts)./sum(counts) .> 0.95)-1]

