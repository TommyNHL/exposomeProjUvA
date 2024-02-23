VERSION
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add("Plots")
#Pkg.add("ProgressBars")
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
#Pkg.add(PackageSpec(url=""))
using Random
using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots
using ProgressBars
#using PyPlot
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#using PyCall, Conda                 #using python packages
#pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier

# inputing 693685 x 4 df
# columns: SMILES, INCHIKEY, PRECURSOR_ION, CNLmasses...
inputDB = CSV.read("F:\\databaseOfInternal_withNLs.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES, :PRECURSOR_ION, :CNLmasses])

# inputing 693685 x 3+1+15961 df
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputCNLs = CSV.read("F:\\dataframeCNLsRows.csv", DataFrame)
sort!(inputCNLs, [:ENTRY])

# creating a table with 2 columns
dfOutput = DataFrame([[],[]], ["INCHIKEY", "FREQUENCY"])
size(dfOutput)

count = 0
str = inputDB[1, "INCHIKEY"]
for i in 1:size(inputDB, 1)
    if (i == size(inputDB, 1))
        temp = []
        count += 1
        push!(temp, inputDB[i, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput, temp)
    elseif (inputDB[i, "INCHIKEY"] == str)
        count += 1
    else
        temp = []
        push!(temp, inputDB[i-1, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput, temp)
        str = inputDB[i, "INCHIKEY"]
        count = 1
    end
end

# 27211 x 2
dfOutput
# save
# output csv is a 27211 x 2 df
savePath = "F:\\countingRows4Leverage.csv"
CSV.write(savePath, dfOutput)

# comparing, 30684 x 793 df
inputAllFPDB = CSV.read("F:\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
sort!(inputAllFPDB, [:INCHIKEY, :SMILES])

# creating a table with 2 columns
dfOutput2 = DataFrame([[],[]], ["INCHIKEY", "FREQUENCY"])
size(dfOutput2)

count = 0
str = inputAllFPDB[1, "INCHIKEY"]
for i in 1:size(inputAllFPDB, 1)
    if (i == size(inputAllFPDB, 1))
        temp = []
        count += 1
        push!(temp, inputAllFPDB[i, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput2, temp)
    elseif (inputAllFPDB[i, "INCHIKEY"] == str)
        count += 1
    else
        temp = []
        push!(temp, inputAllFPDB[i-1, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput2, temp)
        str = inputAllFPDB[i, "INCHIKEY"]
        count = 1
    end
end

# 28536 x 2
dfOutput2
# save
# output csv is a 28536 x 2 df
savePath = "F:\\countingRowsInFP4Leverage.csv"
CSV.write(savePath, dfOutput2)

# creating FP df taking frequency into accounut
# creating a table with 1+FPs+Ri columns
dfOutputFP = DataFrame([[]], ["INCHIKEY"])
for col in names(inputAllFPDB)[3:end]
    dfOutputFP[:, col] = []
end
size(dfOutputFP)  # 0 x 792

for ID in 1:size(dfOutput, 1)
    println(ID)
    for i = 1:dfOutput[ID, "FREQUENCY"]
        temp = []
        push!(temp, dfOutput[ID, "INCHIKEY"])
        rowNo = findall(inputAllFPDB.INCHIKEY .== dfOutput[ID, "INCHIKEY"])[end:end]
        for col in names(inputAllFPDB)[3:end]
            push!(temp, inputAllFPDB[rowNo, col][1])
        end
        push!(dfOutputFP, temp)
    end
end

# 693685 x 1+790+1 df
dfOutputFP
# save
# output csv is a 693685 x 792 df
savePath = "F:\\dataAllFP_withNewPredictedRiWithStratification_Freq.csv"
CSV.write(savePath, dfOutputFP)