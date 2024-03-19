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

# inputing 863582, 865290, 881258, 881919, 876853 x 19 dfs -> 4368902 x 13+4+1 df
    ## ID, NAMEreal, INCHIKEYreal, ACCESSIONreal, MS1Mass, Name, 
    ## RefMatchFrag, UsrMatchFrag, MS1Error, MS2Error, MS2ErrorStd, 
    ## DirectMatch, ReversMatch, FinalScore, 
    ## SpecType, MatchedFrags, INCHIKEY, FragMZ, FragInt
inputDB1 = CSV.read("F:\\Cand_synth_rr10_1_1000.csv", DataFrame)
inputDB2 = CSV.read("F:\\Cand_synth_rr10_1001_2000.csv", DataFrame)
inputDB3 = CSV.read("F:\\Cand_synth_rr10_2001_3000.csv", DataFrame)
inputDB4 = CSV.read("F:\\Cand_synth_rr10_3001_4000.csv", DataFrame)
inputDB5 = CSV.read("F:\\Cand_synth_rr10_4001_5000.csv", DataFrame)
combinedDB = vcat(inputDB1, inputDB2, inputDB3, inputDB4, inputDB5)
combinedDB = combinedDB[:, ["ID", "INCHIKEYreal", "INCHIKEY", 
    "RefMatchFrag", "UsrMatchFrag", 
    "MS1Error", "MS2Error", "MS2ErrorStd", 
    "DirectMatch", "ReversMatch", 
    "FinalScore", "MS1Mass", "FragMZ"]]
combinedDB[!, "RefMatchFragRatio"] .= float(0)
combinedDB[!, "UsrMatchFragRatio"] .= float(0)
combinedDB[!, "FinalScoreRatio"] .= float(0)
combinedDB[!, "INCHIKEY_ID"] .= ""

function takeRatio(str)
    num = ""
    ratio = float(0)
    for i in 1:length(str)
        if num == "-"
            num = ""
            continue
        elseif str[i] == '-'
            ratio = parse(Float64, num)
            #print(ratio)
            num = "-"
        elseif str[i] != '-'
            num = string(num, str[i])
            #print(num)
        end
    end
    ratio = ratio / (parse(Float64, num))
    return ratio
end

takeRatio(combinedDB[4368902, "RefMatchFrag"])
ratioRef = []
ratioUsr = []
ratioScore = []
trueOrFalse = []
for i in 1:size(combinedDB, 1)
    println(i)
    push!(ratioRef, takeRatio(combinedDB[i, "RefMatchFrag"]))
    push!(ratioUsr, takeRatio(combinedDB[i, "UsrMatchFrag"]))
    push!(ratioScore, combinedDB[i, "FinalScore"]/7)
    combinedDB[i, "INCHIKEY_ID"] = string(combinedDB[i, "INCHIKEY"], "_", string(combinedDB[i, "ID"]))
    if (combinedDB[i, "INCHIKEYreal"] == combinedDB[i, "INCHIKEY"])
        push!(trueOrFalse, 1)
    else
        push!(trueOrFalse, 0)
    end
end

combinedDB[!, "RefMatchFragRatio"] = ratioRef
combinedDB[!, "UsrMatchFragRatio"] = ratioUsr
combinedDB[!, "FinalScoreRatio"] = ratioScore
combinedDB[!, "LABEL"] = trueOrFalse

# 4368902 x 18 df
combinedDB

# 4368902 x 3+8+2+1 df
outputDf = combinedDB[:, ["INCHIKEY_ID", "INCHIKEY", "INCHIKEYreal", "RefMatchFragRatio", "UsrMatchFragRatio", 
    "MS1Error", "MS2Error", "MS2ErrorStd", "DirectMatch", "ReversMatch", 
    "FinalScoreRatio", "MS1Mass", "FragMZ", "LABEL"]]

outputDf
# output csv is a 4368902 x 3+8+2+1 df
savePath = "F:\\Cand_synth_rr10_1-5000.csv"
CSV.write(savePath, outputDf)


# creating a 4368902 x 3+8+2+1+1 df, 
    ## columns: INCHIKEY_ID, INCHIKEYreal, 8+1 ULSA features, LABEL
    ##                      FP->Ri, CNL->Ri ^
# matching INCHIKEY, 30684 x 793 df
inputFP2Ri = CSV.read("F:\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
sort!(inputFP2Ri, [:INCHIKEY, :SMILES])

# FP-derived Ri values
outputDf[!, "predictRi"] .= float(0)

for i in 1:size(outputDf, 1)
    println(i)
    if outputDf[i, "INCHIKEY"] in Array(inputFP2Ri[:, "INCHIKEY"])
        rowNo = findall(inputFP2Ri.INCHIKEY .== outputDf[i, "INCHIKEY"])
        outputDf[i, "predictRi"] = inputFP2Ri[rowNo[end:end], "predictRi"][1]
    else
        outputDf[i, "predictRi"] = float(8888888)
    end
end

# 4368902 x 3+8+2+1+1
outputDf
describe(outputDf)
# filtering in INCHIKEY_ID with Ri values
# 4307198 x 15 df
outputDf = outputDf[outputDf.predictRi .!= float(8888888), :]
sort!(outputDf, [:LABEL, :INCHIKEY_ID])

outputDf = outputDf[outputDf.MS2ErrorStd .!== NaN, :]

# output csv is a 4272788 x 3+8+2+1+1 df
savePath = "F:\\Cand_synth_rr10_1-5000_extractedWithoutDeltaRi.csv"
CSV.write(savePath, outputDf)

inputDf = CSV.read("F:\\Cand_synth_rr10_1-5000_extractedWithoutDeltaRi.csv", DataFrame)

testIDsDf = CSV.read("F:\\generated_susp.csv", DataFrame)
testIDsDf = testIDsDf[testIDsDf.ION_MODE .== "POSITIVE", :]

testIDs = Set()
for i in 1:size(testIDsDf, 1)
    push!(testIDs, testIDsDf[i, "INCHIKEY"])
end
distinctTestIDs = sort!(collect(testIDs))

keepIdx = []
isolateIdx = []
for i in 1:size(inputDf, 1)
    println(i)
    if (inputDf[i, "INCHIKEY"] in distinctTestIDs || inputDf[i, "INCHIKEYreal"] in distinctTestIDs)
        push!(isolateIdx, i)
    else
        push!(keepIdx, i)
    end
end

keepIdx  # 4135721
isolateIdx  # 137067

trainValDf = inputDf[keepIdx, :]
# save, ouputing testSet df 4135721 x 15
savePath = "F:\\Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_trainValDf.csv"
CSV.write(savePath, trainValDf)

testDf = inputDf[isolateIdx, :]
# save, ouputing testSet df 137067 x 15
savePath = "F:\\Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_isotestDf.csv"
CSV.write(savePath, testDf)