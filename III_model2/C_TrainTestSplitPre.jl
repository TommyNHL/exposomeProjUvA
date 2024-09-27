## INPUT(S)
# dataframeCNLsRows.csv
# dataAllFP_withNewPredictedRiWithStratification_Freq.csv

## OUTPUT(S)
# databaseOfInternal_withNLsOnly.csv
# databaseOfInternal_withEntryInfoOnly.csv
# databaseOfInternal_withINCHIKEYInfoOnly.csv
# databaseOfInternal_withYOnly.csv

## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using CSV, DataFrames

## input 693685 x 3+1+15961 df ##
# columns: ENTRY, SMILES, INCHIKEY, MONOISOTOPICMASS, CNLmasses...
inputCNLs = CSV.read("F:\\UvA\\dataframeCNLsRows.csv", DataFrame)
sort!(inputCNLs, [:ENTRY])

## extract and save data ##
CNLs = deepcopy(inputCNLs[:, 4:end])
savePath = "F:\\UvA\\databaseOfInternal_withNLsOnly.csv"
CSV.write(savePath, CNLs)

## extract and save index info ##
CNLsInfo = deepcopy(inputCNLs[:, 1:1])
savePath = "F:\\UvA\\databaseOfInternal_withEntryInfoOnly.csv"
CSV.write(savePath, CNLsInfo)


# ==============================================================================
## input a 693685 x 792 df ##
dfOutputFP = CSV.read("F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification_Freq.csv", DataFrame)

## extract and InChIKeys info ##
CNLsInfo2 = deepcopy(dfOutputFP[:, 1:1])
savePath = "F:\\UvA\\databaseOfInternal_withINCHIKEYInfoOnly.csv"
CSV.write(savePath, CNLsInfo2)

## extract and save label info ##
CNLsY = deepcopy(dfOutputFP[:, end:end])
savePath = "F:\\UvA\\databaseOfInternal_withYOnly.csv"
CSV.write(savePath, CNLsY)
