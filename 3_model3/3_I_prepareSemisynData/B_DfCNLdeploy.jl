## INPUT(S)
# Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_trainValDf.csv or Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_isotestDf.csv
# dataframe73_dfTestSetWithStratification_withCNLPredictedRi.csv
# CocamideExtended73_CNLsRi_RFwithStratification.joblib

## OUTPUT(S)
# TPTN_dfCNLfeaturesStr.csv
# Cand_synth_rr10_1-5000_extractedWithCNLsList.csv or Cand_synth_rr10_1-5000_extractedWithCNLsList_pest.csv
# dfCNLsSum_1.csv - dfCNLsSum_8.csv or dfCNLsSum_pest.csv
# TPTNmassesCNLsDistrution_1.png - TPTNmassesCNLsDistrution_8.png
# dataframeCNLsRows4TPTNModeling_1withCNLRideltaRi.csv - dataframeCNLsRows4TPTNModeling_8withCNLRideltaRi.csv or dataframeCNLsRows4TPTNModeling_PestwithCNLRideltaRi.csv
# dataframeCNLsRows4TPTNModeling_TPOnlywithCNLRideltaRi.csv or dataframeCNLsRows4TPTNModeling_TPOnlywithCNLRideltaRi_pest.csv
# dfCNLsSum_TP.csv or dfCNLsSum_TP_pest.csv
# dfCNLsSum.csv
# TPTNmassesCNLsDistrution.png or TPTNmassesCNLsDistrution_pest.png

VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
#Pkg.add(PackageSpec(url=""))
using Random
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots

## import packages from Python ##
jl = pyimport("joblib")             # used for loading models
using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

## input 4135721 x 3+8+2+1+1 df ##
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputTPTNdf = CSV.read("F:\\Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_trainValDf.csv", DataFrame)
# inputing 137067 x 3+8+2+1+1 df
#inputTPTNdf = CSV.read("F:\\Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_isotestDf.csv", DataFrame)
sort!(inputTPTNdf, [:LABEL, :INCHIKEY_ID])

## define a function for data extraction ##
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

## initialize an array for 1 more column ## -> 4135721 x 15+1 or 137067 x 15+1
inputTPTNdf[!, "CNLmasses"] .= [[]]
size(inputTPTNdf)

## calculate neutral loss masses (NLs) ##
## filter in CNL masses-of-interest, and store in Vector{Any} ##
    ## filter in CNLs feature ##
    # according to the pre-defined CNLs in CNLs_10mDa.csv
    # inputing 15961 candidates
    candidates_df = CSV.read("F:\\UvA\\dataframe73_dfTestSetWithStratification_withCNLPredictedRi.csv", DataFrame)
    CNLfeaturesStr = names(candidates_df)[4:end-2]

    ## make string
    dfCNLfeaturesStr = DataFrame([[]], ["CNLfeaturesStr"])
    for CNLfeature in CNLfeaturesStr
        list = [CNLfeature]
        push!(dfCNLfeaturesStr, list)
    end

    ## save features in string ##
    savePath = "F:\\UvA\\TPTN_dfCNLfeaturesStr.csv"
    CSV.write(savePath, dfCNLfeaturesStr)

CNLfeaturesStr = CSV.read("F:\\TPTN_dfCNLfeaturesStr.csv", DataFrame)[:, "CNLfeaturesStr"]
#
candidatesList = []
for can in CNLfeaturesStr
    #push!(candidatesList, round(parse(Float64, can), digits = 2))
    push!(candidatesList, round(can, digits = 2))
end
CNLfeaturesStr = []
for can in candidatesList
    #push!(candidatesList, round(parse(Float64, can), digits = 2))
    push!(CNLfeaturesStr, string(can))
end
#
for i in 1:size(inputTPTNdf, 1)
    println(i)
    fragIons = getVec(inputTPTNdf[i,"FragMZ"])
    arrNL = Set()
    for frag in fragIons
        if (inputTPTNdf[i,"MS1Mass"] - frag) >= float(0)
            NL = round((inputTPTNdf[i,"MS1Mass"] - frag), digits = 2)
            if (NL in candidatesList)
                push!(arrNL, NL)
            end
        end
    end
    inputTPTNdf[i, "CNLmasses"] = sort!(collect(arrNL))
end
sort!(inputTPTNdf, [:LABEL, :INCHIKEY_ID, :CNLmasses])

## reduce df size (rows) ##
function getMasses(db, i, arr, arrType = "str")
    massesArr = arr
    if (arrType == "dig")
        masses = db[i, "CNLmasses"]
    elseif (arrType == "str")
        masses = getVec(db[i, "CNLmasses"])
    end
    for mass in masses
        push!(massesArr, mass)
    end
    return massesArr
end
#
    ## remove rows that has Frag-ion of interest < 2 (optional)
    retain = []
    for i in 1:size(inputTPTNdf, 1)
        println(i)
        arr = []
        arr = getMasses(inputTPTNdf, i, arr, "dig")#"str")
        if (size(arr, 1) >= 2)
            push!(retain, i)
        end
    end
    inputTPTNdf = inputTPTNdf[retain, :]
## save ##
savePath = "F:\\Cand_synth_rr10_1-5000_extractedWithCNLsList.csv"
#savePath = "F:\\UvA\\Cand_synth_rr10_1-5000_extractedWithCNLsList_pest.csv"
CSV.write(savePath, inputTPTNdf)

## read ##
# 4103848 x 16 df
inputTPTNdf = CSV.read("F:\\Cand_synth_rr10_1-5000_extractedWithCNLsList.csv", DataFrame)
#
# a table with 13+15961+1 columns features CNLs
CNLfeaturesStr
candidatesList
inputTPTNdf
#
## store data in a Matrix ##
X = zeros(512981, 15961)
#X = zeros(136678, 15961)
#
for i in (1+512981*0):(512981*1)
#for i in 1:136678
    println(i)
    arr = []
    arr = getMasses(inputTPTNdf, i, arr, "dig")#"str")
    mumIon = round(inputTPTNdf[i, "MS1Mass"], digits = 2)
    for col in arr
        mz = findall(x->x==col, candidatesList)
        if (col <= mumIon)
            X[i-(512981*0), mz] .= 1
        elseif (col > mumIon)
            X[i-(512981*0), mz] .= -1
        end
    end
end
#
# 4103848 - 512981
dfCNLs = DataFrame(X, CNLfeaturesStr)
insertcols!(dfCNLs, 1, ("ENTRY"=>collect((1+512981*0):(512981*1))))
insertcols!(dfCNLs, 2, ("INCHIKEY_ID"=>inputTPTNdf[(1+512981*0):(512981*1), "INCHIKEY_ID"]))
insertcols!(dfCNLs, 3, ("INCHIKEY"=>inputTPTNdf[(1+512981*0):(512981*8), "INCHIKEY"]))
insertcols!(dfCNLs, 4, ("INCHIKEYreal"=>inputTPTNdf[(1+512981*7):(512981*8), "INCHIKEYreal"]))
insertcols!(dfCNLs, 5, ("RefMatchFragRatio"=>inputTPTNdf[(1+512981*7):(512981*8), "RefMatchFragRatio"]))
insertcols!(dfCNLs, 6, ("UsrMatchFragRatio"=>inputTPTNdf[(1+512981*7):(512981*8), "UsrMatchFragRatio"]))
insertcols!(dfCNLs, 7, ("MS1Error"=>inputTPTNdf[(1+512981*7):(512981*8), "MS1Error"]))
insertcols!(dfCNLs, 8, ("MS2Error"=>inputTPTNdf[(1+512981*7):(512981*8), "MS2Error"]))
insertcols!(dfCNLs, 9, ("MS2ErrorStd"=>inputTPTNdf[(1+512981*7):(512981*8), "MS2ErrorStd"]))
insertcols!(dfCNLs, 10, ("DirectMatch"=>inputTPTNdf[(1+512981*7):(512981*8), "DirectMatch"]))
insertcols!(dfCNLs, 11, ("ReversMatch"=>inputTPTNdf[(1+512981*7):(512981*8), "ReversMatch"]))
insertcols!(dfCNLs, 12, ("FinalScoreRatio"=>inputTPTNdf[(1+512981*7):(512981*8), "FinalScoreRatio"]))
insertcols!(dfCNLs, 13, ("MONOISOTOPICMASS"=>((inputTPTNdf[(1+512981*7):(512981*8), "MS1Mass"] .- 1.007276)/1000)))
dfCNLs[!, "FPpredictRi"] = inputTPTNdf[(1+512981*7):(512981*8), "predictRi"]
size(dfCNLs)  # 512981 x (13+15961+1)

# for pesticide dataset, 136678
#= dfCNLs = DataFrame(X, CNLfeaturesStr)
insertcols!(dfCNLs, 1, ("ENTRY"=>collect(1:136678)))
insertcols!(dfCNLs, 2, ("INCHIKEY_ID"=>inputTPTNdf[1:136678, "INCHIKEY_ID"]))
insertcols!(dfCNLs, 3, ("INCHIKEY"=>inputTPTNdf[1:136678, "INCHIKEY"]))
insertcols!(dfCNLs, 4, ("INCHIKEYreal"=>inputTPTNdf[1:136678, "INCHIKEYreal"]))
insertcols!(dfCNLs, 5, ("RefMatchFragRatio"=>inputTPTNdf[1:136678, "RefMatchFragRatio"]))
insertcols!(dfCNLs, 6, ("UsrMatchFragRatio"=>inputTPTNdf[1:136678, "UsrMatchFragRatio"]))
insertcols!(dfCNLs, 7, ("MS1Error"=>inputTPTNdf[1:136678, "MS1Error"]))
insertcols!(dfCNLs, 8, ("MS2Error"=>inputTPTNdf[1:136678, "MS2Error"]))
insertcols!(dfCNLs, 9, ("MS2ErrorStd"=>inputTPTNdf[1:136678, "MS2ErrorStd"]))
insertcols!(dfCNLs, 10, ("DirectMatch"=>inputTPTNdf[1:136678, "DirectMatch"]))
insertcols!(dfCNLs, 11, ("ReversMatch"=>inputTPTNdf[1:136678, "ReversMatch"]))
insertcols!(dfCNLs, 12, ("FinalScoreRatio"=>inputTPTNdf[1:136678, "FinalScoreRatio"]))
insertcols!(dfCNLs, 13, ("MONOISOTOPICMASS"=>((inputTPTNdf[1:136678, "MS1Mass"] .- 1.007276)/1000)))
dfCNLs[!, "FPpredictRi"] = inputTPTNdf[1:136678, "predictRi"]
size(dfCNLs)  # 512981 x (13+15961+1) =#

#################################################################################
## plot histogram ##
desStat = describe(dfCNLs)  # 15975 x 7
desStat[14,:]
#
sumUp = []
push!(sumUp, 8888888)
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
for col in names(dfCNLs)[14:end-1]
    count = 0
    for i in 1:size(dfCNLs, 1)
        count += dfCNLs[i, col]
    end
    push!(sumUp, count)
end
push!(sumUp, 8888888)
push!(dfCNLs, sumUp)
# 512981 -> 512982 rows
dfCNLsSum = dfCNLs[end:end, :]
savePath = "F:\\UvA\\dfCNLsSum_8.csv"
#savePath = "F:\\UvA\\dfCNLsSum_pest.csv"
CSV.write(savePath, dfCNLsSum)
#
using DataSci4Chem
massesCNLsDistrution = bar(candidatesList, Vector(dfCNLs[end, 14:end-1]), 
    label = false, 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    xtickfontsize = 12, 
    ytickfontsize= 12, 
    xlabel="Feature CNL mass", xguidefontsize=16, 
    ylabel="Count", yguidefontsize=16, 
    dpi = 300)
## save figure ##
savefig(massesCNLsDistrution, "F:\\UvA\\TPTNmassesCNLsDistrution_8.png")


#################################################################################
## predict CNL-derived Ri ##
dfCNLs = dfCNLs[1:end-1, :]
    ## load a model ##
    # requires python 3.11 or 3.12
    modelRF_CNL = jl.load("F:\\UvA\\CocamideExtended73_CNLsRi_RFwithStratification.joblib")
    size(modelRF_CNL)
CNLpredictedRi = predict(modelRF_CNL, Matrix(dfCNLs[:, 13:end-1]))
dfCNLs[!, "CNLpredictRi"] = CNLpredictedRi
dfCNLs[!, "DeltaRi"] = (CNLpredictedRi - dfCNLs[:, "FPpredictRi"]) / 1000
dfCNLs[!, "LABEL"] = inputTPTNdf[(1+512981*7):(512981*8), "LABEL"]
dfCNLs[!, "LABEL"] = inputTPTNdf[1:136678, "LABEL"]

## save ##
savePath = "F:\\UvA\\dataframeCNLsRows4TPTNModeling_8withCNLRideltaRi.csv"
CSV.write(savePath, dfCNLs)
println("done for saving csv")

    modelRF_CNL = jl.load("F:\\UvA\\CocamideExtended73_CNLsRi_RFwithStratification.joblib")
    size(modelRF_CNL)
CNLpredictedRi = predict(modelRF_CNL, Matrix(dfCNLs[:, 13:end-4]))
dfCNLs[:, "CNLpredictRi"] = CNLpredictedRi
dfCNLs[:, "DeltaRi"] = (CNLpredictedRi - dfCNLs[:, "FPpredictRi"]) / 1000
describe(dfCNLs[:, end-4:end])
## save ##
savePath = "F:\\UvA\\dataframeCNLsRows4TPTNModeling_PestwithCNLRideltaRi.csv"
CSV.write(savePath, dfCNLs)
println("done for saving csv")


#################################################################################
## plot histogram ##
outputDf = (dfCNLs[dfCNLs.LABEL .== 1, :])[:, 1:end-3]
savePath = "F:\\UvA\\dataframeCNLsRows4TPTNModeling_TPOnlywithCNLRideltaRi.csv"
savePath = "F:\\UvA\\dataframeCNLsRows4TPTNModeling_TPOnlywithCNLRideltaRi_pest.csv"
CSV.write(savePath, outputDf)
println("done for saving csv")
#
sumUp = []
push!(sumUp, 8888888)
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
for col in names(outputDf)[14:end-1]
    count = 0
    for i in 1:size(outputDf, 1)
        count += outputDf[i, col]
    end
    push!(sumUp, count)
end
push!(sumUp, 8888888)
push!(outputDf, sumUp)
# + 1 rows
dfCNLsSum = outputDf[end:end, :]
savePath = "F:\\UvA\\dfCNLsSum_TP.csv"
#savePath = "F:\\UvA\\dfCNLsSum_TP_pest.csv"
CSV.write(savePath, dfCNLsSum)
#
dfCNLsSum1 = CSV.read("F:\\UvA\\dfCNLsSum_1.csv", DataFrame)
dfCNLsSum2 = CSV.read("F:\\UvA\\dfCNLsSum_2.csv", DataFrame)
dfCNLsSum3 = CSV.read("F:\\UvA\\dfCNLsSum_3.csv", DataFrame)
dfCNLsSum4 = CSV.read("F:\\UvA\\dfCNLsSum_4.csv", DataFrame)
dfCNLsSum5 = CSV.read("F:\\UvA\\dfCNLsSum_5.csv", DataFrame)
dfCNLsSum6 = CSV.read("F:\\UvA\\dfCNLsSum_6.csv", DataFrame)
dfCNLsSum7 = CSV.read("F:\\UvA\\dfCNLsSum_7.csv", DataFrame)
dfCNLsSum8 = CSV.read("F:\\UvA\\dfCNLsSum_8.csv", DataFrame)
dfCNLsSum = vcat(dfCNLsSum1, dfCNLsSum2, dfCNLsSum3, dfCNLsSum4, dfCNLsSum5, dfCNLsSum6, dfCNLsSum7, dfCNLsSum8)
dfCNLsSumTP = CSV.read("F:\\UvA\\dfCNLsSum_TP.csv", DataFrame)
#
# for pesticide dataset only
#= dfCNLsSum = CSV.read("F:\\UvA\\dfCNLsSum_pest.csv", DataFrame)
dfCNLsSumTP = CSV.read("F:\\UvA\\dfCNLsSum_TP_pest.csv", DataFrame) =#
#
sumUp = []
push!(sumUp, 8888888)
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
for col in names(dfCNLsSum)[14:end-1]
    count = 0
    for i in 1:size(dfCNLsSum, 1)
        count += dfCNLsSum[i, col]
    end
    push!(sumUp, count)
end
push!(sumUp, 8888888)
push!(dfCNLsSum, sumUp)
# 225572 -> 225573 rows
dfCNLsSum = dfCNLsSum[end:end, :]
savePath = "F:\\UvA\\dfCNLsSum.csv"
CSV.write(savePath, dfCNLsSum)
#
using DataSci4Chem
massesCNLsDistrution = bar(candidatesList, Vector(dfCNLsSum[end, 14:end-1]), 
    label = "True negative CNLs", 
    fc = "pink", 
    lc = "pink", 
    margin = (5, :mm), 
    size = (1000,800), 
    xtickfontsize = 12, 
    ytickfontsize= 12, 
    xlabel="Feature CNL mass", xguidefontsize=16, 
    ylabel="Count", yguidefontsize=16, 
    legendfont = font(12), 
    dpi = 300)
    bar!(candidatesList, Vector(dfCNLsSumTP[end, 14:end-1]), 
        label = "True positive CNLs", 
        fc = "skyblue", 
        lc = "skyblue", 
        margin = (5, :mm), 
        size = (1000,800), 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        xlabel="Feature CNL mass", xguidefontsize=16, 
        ylabel="Count", yguidefontsize=16, 
        legendfont = font(12), 
        dpi = 300)
## save figure ##
savefig(massesCNLsDistrution, "F:\\UvA\\TPTNmassesCNLsDistrution.png")
savefig(massesCNLsDistrution, "F:\\UvA\\TPTNmassesCNLsDistrution_pest.png")
