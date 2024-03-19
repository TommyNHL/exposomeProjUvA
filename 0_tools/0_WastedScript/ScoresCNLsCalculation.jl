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

# inputing 2521 x 4 df
# columns: SMILES, INCHIKEY, 20494 CNLs
inputDB = CSV.read("D:\\0_data\\dataframeCNLsTPTN.csv", DataFrame)
dfP = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
for col in names(inputDB)[3:end]
    dfP[:, col] = []
end
dfP[!, "compoundCNLsSUM"] .= 0
size(dfP)

# calculating SUM(P(TP)) for each CNL
# calculating SUM(P(TN)) for each CNL
for i in 1:size(inputDB, 1)
    println(i)
    count = 0
    rowP = []
    push!(rowP, inputDB[i, "SMILES"])
    push!(rowP, inputDB[i, "INCHIKEY"])
    for col in names(inputDB)[3:end]
        count += inputDB[i, col]
    end
    for col in names(inputDB)[3:end]
        temp = inputDB[i, col] / count
        push!(rowP, temp)
    end
    push!(rowP, count)
    push!(dfP, rowP)
end
size(dfP)
Array(dfP[3, 1:end])
Array(dfP[3, 3:end])

# ouputing df 5042 x 20497
savePath = "D:\\0_data\\dataframePofCNLsTPTN.csv"
CSV.write(savePath, dfP)

# calculating Score(CNL) for each CNL = 1 - SUM(P(TP)) / SUM(P(TN))
dfScoresCNL = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
for col in names(inputDB)[3:end]
    dfScoresCNL[:, col] = []
end
size(dfScoresCNL)

for i in 1:size(dfP, 1)
    if (i % 2 == 0)
        continue
    elseif (i % 2 == 1)
        println(i)
        count = 0
        scoresCNLs = []
        push!(scoresCNLs, inputDB[i, "SMILES"])
        push!(scoresCNLs, inputDB[i, "INCHIKEY"])
        for col in names(dfP)[3:end-1]
            score = 1 - (dfP[i, col] / dfP[i+1, col])
            push!(scoresCNLs, score)
        end
        push!(dfScoresCNL, scoresCNLs)
     end
end
size(dfScoresCNL)
Array(dfScoresCNL[3, 1:end])
Array(dfScoresCNL[3, 3:end])

# ouputing df 2521 x 20496
savePath = "D:\\0_data\\dataframeScoresCNLsTPTNwithoutSelection.csv"
CSV.write(savePath, dfScoresCNL)