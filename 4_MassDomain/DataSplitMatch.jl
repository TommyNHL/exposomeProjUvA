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

# inputing 32862, 38187, 35975 x 19 dfs -> 107024 x 11+1+1 df
    ## ID, NAMEreal, INCHIKEYreal, ACCESSIONreal, MS1Mass, Name, 
    ## RefMatchFrag, UsrMatchFrag, MS1Error, MS2Error, MS2ErrorStd, 
    ## DirectMatch, ReversMatch, FinalScore, 
    ## SpecType, MatchedFrags, INCHIKEY, FragMZ, FragInt
inputDB1 = CSV.read("D:\\Cand_search_rr0_0612_TEST_100-200.csv", DataFrame)
inputDB2 = CSV.read("D:\\Cand_search_rr0_0612_TEST_200-300.csv", DataFrame)
inputDB3 = CSV.read("D:\\Cand_search_rr0_0612_TEST_300-400.csv", DataFrame)
combinedDB = vcat(inputDB1, inputDB2, inputDB3)
combinedDB = combinedDB[:, ["ID", "INCHIKEYreal", "INCHIKEY", 
    "RefMatchFrag", "UsrMatchFrag", 
    "MS1Error", "MS2Error", "MS2ErrorStd", 
    "DirectMatch", "ReversMatch", 
    "FinalScore"]]
combinedDB[!, "INCHIKEY_ID"] .= ""

trueOrFalse = []
for i in 1:size(combinedDB, 1)
    combinedDB[i, "INCHIKEY_ID"] = string(combinedDB[i, "INCHIKEY"], "_", string(combinedDB[i, "ID"]))
    if (combinedDB[i, "INCHIKEYreal"] == combinedDB[i, "INCHIKEY"])
        push!(trueOrFalse, 1)
    else
        push!(trueOrFalse, 0)
    end
end

combinedDB[!, "LABEL"] = trueOrFalse

# 107024 x 11+1+1 df
combinedDB

outputDf = combinedDB[:, ["INCHIKEY_ID", "RefMatchFrag", "UsrMatchFrag", 
    "MS1Error", "MS2Error", "MS2ErrorStd", "DirectMatch", "ReversMatch", 
    "FinalScore", "LABEL"]]

outputDf[1:5,1]

# output csv is a 107024 x 1+8+1 df
savePath = "D:\\Cand_search_rr0_0612_TEST_100-400_extracted.csv"
CSV.write(savePath, outputDf)

# creating a 107024 x 1+8+2+1+1 df, 
    ## columns: INCHIKEY_ID, 8+1 ULSA features, LABEL
    ## added columns: FP->Ri, CNL->Ri, deltaRi ^
# matching INCHIKEY, 30684 x 793 df
inputAllRi_TrainSet = CSV.read("D:\\0_data\\dataframe_dfTrainSetWithStratification_withCNLPredictedRi_withCocamides.csv", DataFrame)
sort!(inputAllRi_TrainSet, [:INCHIKEY, :SMILES])

# matching INCHIKEY, 30684 x 793 df
inputAllRi_TestSet = CSV.read("D:\\0_data\\dataframe_dfTestSetWithStratification_withCNLPredictedRi_withCocamides.csv", DataFrame)
sort!(inputAllRi_TestSet, [:INCHIKEY, :SMILES])

outputDf[!, "predictRi"] = float(0)
outputDf[!, "CNLpredictRi"] = float(0)
outputDf[!, "DeltaRi"] = float(0)
for i in 1:size(outputDf, 1)
    if outputDf[i, "INCHIKEY"] in Array(inputAllRi_TrainSet[:, "INCHIKEY"])
        rowNo = findall(inputAllRi_TrainSet.INCHIKEY .== outputDf[i, "INCHIKEY"])
        outputDf[!, "predictRi"] = inputAllRi_TrainSet[rowNo[end:end], "predictRi"][1]
        outputDf[!, "CNLpredictRi"] = inputAllRi_TrainSet[rowNo[end:end], "CNLpredictRi"][1]
        outputDf[!, "DeltaRi"] = outputDf[!, "CNLpredictRi"] - outputDf[!, "predictRi"]
    elseif outputDf[i, "INCHIKEY"] in Array(inputAllRi_TestSet[:, "INCHIKEY"])
        rowNo = findall(inputAllRi_TestSet.INCHIKEY .== outputDf[i, "INCHIKEY"])
        outputDf[!, "predictRi"] = inputAllRi_TestSet[rowNo[end:end], "predictRi"][1]
        outputDf[!, "CNLpredictRi"] = inputAllRi_TestSet[rowNo[end:end], "CNLpredictRi"][1]
        outputDf[!, "DeltaRi"] = outputDf[!, "CNLpredictRi"] - outputDf[!, "predictRi"]
    else
        outputDf[!, "predictRi"] = float(888888)
        outputDf[!, "CNLpredictRi"] = float(888888)
        outputDf[!, "DeltaRi"] = float(0)
    end
end

# filtering in INCHIKEY_ID with Ri values
outputDf = outputDf[outputDf.predictRi .!= float(888888), ["INCHIKEY_ID", 
    "RefMatchFrag", "UsrMatchFrag", "MS1Error", "MS2Error", "MS2ErrorStd", 
    "DirectMatch", "ReversMatch", "FinalScore", "DeltaRi", "LABEL"]]
sort!(outputDf, [:LABEL, :INCHIKEY_ID, :DeltaRi, :FinalScore])

# output csv is a xxx x 1+8+1+1 df
savePath = "D:\\Cand_search_rr0_0612_TEST_100-400_extractedWithDeltaRi.csv"
CSV.write(savePath, outputDf)