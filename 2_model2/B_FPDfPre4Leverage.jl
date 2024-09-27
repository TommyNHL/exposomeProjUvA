## INPUT(S)
# databaseOfInternal_withNLs.csv
# dataframeCNLsRows.csv
# dataAllFP_withNewPredictedRiWithStratification.csv

## OUTPUT(S)
# countingRows4Leverage.csv
# countingRowsInFP4Leverage.csv
# dataAllFP_withNewPredictedRiWithStratification_Freq.csv

## install packages needed ##
using Pkg
#Pkg.add("PyCall")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using Random
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using ProgressBars

## input 693685 x 4 df ##
# columns: SMILES, INCHIKEY, PRECURSOR_ION, CNLmasses...
inputDB = CSV.read("F:\\UvA\\databaseOfInternal_withNLs.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES, :PRECURSOR_ION, :CNLmasses])

## input 693685 x 3+1+15961 df ##
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputCNLs = CSV.read("F:\\UvA\\dataframeCNLsRows.csv", DataFrame)
sort!(inputCNLs, [:ENTRY])

## create a table with 2 columns ##
dfOutput = DataFrame([[],[]], ["INCHIKEY", "FREQUENCY"])
size(dfOutput)

## count number of replicates for individual InChIKey ##
# for leverage calculation
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
## save information ##
# output csv is a 27211 x 2 df
savePath = "F:\\UvA\\countingRows4Leverage.csv"
CSV.write(savePath, dfOutput)

# compare with all InChIKeys in the internal DB, 30684 x 793 df
inputAllFPDB = CSV.read("F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
sort!(inputAllFPDB, [:INCHIKEY, :SMILES])
    # create a table with 2 columns
    dfOutput2 = DataFrame([[],[]], ["INCHIKEY", "FREQUENCY"])
    size(dfOutput2)
    #
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
# save informaiton ##
# output csv is a 28536 x 2 df
savePath = "F:\\UvA\\countingRowsInFP4Leverage.csv"
CSV.write(savePath, dfOutput2)

## copy FP df taking frequency into accounut ##
# create a table with 1+FPs+Ri columns ##
dfOutputFP = DataFrame([[]], ["INCHIKEY"])
for col in names(inputAllFPDB)[3:end]
    dfOutputFP[:, col] = []
end
size(dfOutputFP)  # 0 x 792
#
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
## save output df as a spreadsheet ##
# output csv is a 693685 x 792 df
savePath = "F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification_Freq.csv"
CSV.write(savePath, dfOutputFP)
