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

# inputing 863582, 865290, 881258, 881919, 876853 x 19 dfs -> 4368902 x 2+13+3+1 df
    ## ID, NAMEreal, INCHIKEYreal, ACCESSIONreal, MS1Mass, Name, 
    ## RefMatchFrag, UsrMatchFrag, MS1Error, MS2Error, MS2ErrorStd, 
    ## DirectMatch, ReversMatch, FinalScore, 
    ## SpecType, MatchedFrags, INCHIKEY, FragMZ, FragInt
inputDB1 = CSV.read("D:\\Cand_synth_rr10_1_1000.csv", DataFrame) 
inputDB2 = CSV.read("D:\\Cand_synth_rr10_1001_2000.csv", DataFrame)
inputDB3 = CSV.read("D:\\Cand_synth_rr10_2001_3000.csv", DataFrame)
inputDB4 = CSV.read("D:\\Cand_synth_rr10_3001_4000.csv", DataFrame)
inputDB5 = CSV.read("D:\\Cand_synth_rr10_4001_5000.csv", DataFrame)
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

function getVec(matStr)
    if matStr[1] .== '['
        if contains(matStr, ", ")
            str = split(matStr[2:end-1],", ")
        else
            str = split(matStr[2:end-1]," ")
        end
    elseif matStr[1] .== 'A'
        if contains(matStr, ", ")
            str = split(matStr[5:end-1],", ")
        else
            str = split(matStr[5:end-1]," ")
        end
    elseif matStr[1] .== 'F'
        if matStr .== "Float64[]"
            return []
        else
            str = split(matStr[9:end-1],", ")
        end
    elseif matStr[1] .== 'I'
        if matStr .== "Int64[]"
            return []
        else
            str = split(matStr[7:end-1],", ")
        end
    else
        println("New vector start")
        println(matStr)
    end
    if length(str) .== 1 && cmp(str[1],"") .== 0
        return []
    else
        str = parse.(Float64, str)
        return str
    end
end

ratioRef = []
ratioUsr = []
ratioScore = []
trueOrFalse = []
for i in 1:size(combinedDB, 1)
    println(i)
    push!(ratioRef, takeRatio(combinedDB[i, "RefMatchFragRatio"]))
    push!(ratioUsr, takeRatio(combinedDB[i, "UsrMatchFragRatio"]))
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

combinedDB

outputDf = combinedDB[:, ["INCHIKEY_ID", "INCHIKEY", "RefMatchFragRatio", "UsrMatchFragRatio", 
    "MS1Error", "MS2Error", "MS2ErrorStd", "DirectMatch", "ReversMatch", 
    "FinalScoreRatio", "MS1Mass", "FragMZ", "LABEL"]]

outputDf
# output csv is a 107024 x 1+8+1 df
savePath = "D:\\Cand_synth_rr10_1_5000.csv"
CSV.write(savePath, outputDf)


# creating a 107024 x 1+8+2+1+1 df, 
    ## columns: INCHIKEY_ID, 8+1 ULSA features, LABEL
    ## added columns: FP->Ri, CNL->Ri, deltaRi ^
# matching INCHIKEY, 30684 x 793 df
inputFP2Ri = CSV.read("D:\\0_data\\dataAllFP_withNewPredictedRi.csv", DataFrame)
sort!(inputFP2Ri, [:INCHIKEY, :SMILES])

outputDf[!, "predictRi"] .= float(0)
outputDf[!, "CNLpredictRi"] .= float(0)
outputDf[!, "DeltaRi"] .= float(0)


for i in 1:size(outputDf, 1)
    if outputDf[i, "INCHIKEY"] in Array(inputFP2Ri[:, "INCHIKEY"])
        rowNo = findall(inputFP2Ri.INCHIKEY .== outputDf[i, "INCHIKEY"])
        outputDf[i, "predictRi"] = inputFP2Ri[rowNo[end:end], "predictRi"][1]
        #outputDf[!, "CNLpredictRi"] = inputAllRi_TrainSet[rowNo[end:end], "CNLpredictRi"][1]
        #outputDf[!, "DeltaRi"] = outputDf[!, "CNLpredictRi"] - outputDf[!, "predictRi"]
    elseif outputDf[i, "INCHIKEY"] in Array(inputFP2Ri[:, "INCHIKEY"])
        rowNo = findall(inputFP2Ri.INCHIKEY .== outputDf[i, "INCHIKEY"])
        outputDf[!, "predictRi"] = inputFP2Ri[rowNo[end:end], "predictRi"][1]
        #outputDf[!, "CNLpredictRi"] = inputAllRi_TestSet[rowNo[end:end], "CNLpredictRi"][1]
        #outputDf[!, "DeltaRi"] = outputDf[!, "CNLpredictRi"] - outputDf[!, "predictRi"]
    else
        outputDf[i, "predictRi"] = float(888888)
        #outputDf[!, "CNLpredictRi"] = float(888888)
        #outputDf[!, "DeltaRi"] = float(0)
    end
end

outputDf
describe(outputDf[:, "predictRi"])
# filtering in INCHIKEY_ID with Ri values
outputDf = outputDf[outputDf.predictRi .!= float(888888), :]
sort!(outputDf, [:LABEL, :INCHIKEY_ID])

# output csv is a xxx x 1+8+1+1 df
savePath = "D:\\Cand_synth_rr10_1_5000_extractedWithoutDeltaRi.csv"
CSV.write(savePath, outputDf)