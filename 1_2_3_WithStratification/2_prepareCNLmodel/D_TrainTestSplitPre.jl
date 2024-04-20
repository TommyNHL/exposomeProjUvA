using Pkg
using CSV, DataFrames

# inputing 693685 x 3+1+15961 df
# columns: ENTRY, SMILES, INCHIKEY, MONOISOTOPICMASS, CNLmasses...
inputCNLs = CSV.read("F:\\UvA\\dataframeCNLsRows.csv", DataFrame)
sort!(inputCNLs, [:ENTRY])

CNLs = deepcopy(inputCNLs[:, 4:end])
savePath = "F:\\UvA\\databaseOfInternal_withNLsOnly.csv"
CSV.write(savePath, CNLs)

CNLsInfo = deepcopy(inputCNLs[:, 1:1])
savePath = "F:\\UvA\\databaseOfInternal_withEntryInfoOnly.csv"
CSV.write(savePath, CNLsInfo)


# ==============================================================================
# inputing a 693685 x 792 df
dfOutputFP = CSV.read("F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification_Freq.csv", DataFrame)
CNLsInfo2 = deepcopy(dfOutputFP[:, 1:1])
savePath = "F:\\UvA\\databaseOfInternal_withINCHIKEYInfoOnly.csv"
CSV.write(savePath, CNLsInfo2)

CNLsY = deepcopy(dfOutputFP[:, end:end])
savePath = "F:\\UvA\\databaseOfInternal_withYOnly.csv"
CSV.write(savePath, CNLsY)