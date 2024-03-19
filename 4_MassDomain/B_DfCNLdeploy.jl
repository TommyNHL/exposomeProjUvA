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
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

# inputing 4135721 x 3+8+2+1+1 df
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputTPTNdf = CSV.read("F:\\Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_trainValDf.csv", DataFrame)
sort!(inputTPTNdf, [:LABEL, :INCHIKEY_ID])

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


inputTPTNdf

inputTPTNdf[1, "MS1Mass"]
inputTPTNdf[1, "FragMZ"]

# initialization for 1 more column -> 4272788 x 15+1
inputTPTNdf[!, "CNLmasses"] .= [[]]
size(inputTPTNdf)

# NLs calculation, filtering CNL-in-interest, storing in Vector{Any}
        # filtering in CNLs features according to the pre-defined CNLs in CNLs_10mDa.csv
        # inputing 15961 candidates
        candidates_df = CSV.read("F:\\dataframe_dfTrainSetWithStratification_withCNLPredictedRi.csv", DataFrame)
        CNLfeaturesStr = names(candidates_df)[4:end-2]

        dfCNLfeaturesStr = DataFrame([[]], ["CNLfeaturesStr"])
        for CNLfeature in CNLfeaturesStr
            list = [CNLfeature]
            push!(dfCNLfeaturesStr, list)
        end

        savePath = "F:\\TPTN_dfCNLfeaturesStr.csv"
        CSV.write(savePath, dfCNLfeaturesStr)

        CNLfeaturesStr = CSV.read("F:\\TPTN_dfCNLfeaturesStr.csv", DataFrame)[:, "CNLfeaturesStr"]

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
inputTPTNdf[:, "CNLmasses"]

savePath = "F:\\Cand_synth_rr10_1-5000_extractedWithCNLsList.csv"
CSV.write(savePath, inputTPTNdf)

# 4135721 x 16 df
inputTPTNdf = CSV.read("F:\\Cand_synth_rr10_1-5000_extractedWithCNLsList.csv", DataFrame)

# Reducing df size (rows)
        function getMasses(db, i, arr)
            massesArr = arr
            masses = getVec(db[i, "CNLmasses"])
            for mass in masses
                push!(massesArr, mass)
            end
            return massesArr
        end


# creating a table with 13+15961+1 columns features CNLs
CNLfeaturesStr
candidatesList
inputTPTNdf

# storing data in a Matrix
X = zeros(516959, 15961)

#for i in 1:1076799
for i in (1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959)
    println(i)
    arr = []
    arr = getMasses(inputTPTNdf, i, arr)
    mumIon = round(inputTPTNdf[i, "MS1Mass"], digits = 2)
    for col in arr
        mz = findall(x->x==col, candidatesList)
        if (col <= mumIon)
            #X[i, mz] .= 1
            X[i-516966-516966-516966-516966-516966-516966-516966, mz] .= 1
        elseif (col > mumIon)
            #X[i, mz] .= -1
            X[i-516966-516966-516966-516966-516966-516966-516966, mz] .= -1
        end
    end
end

# 4135721 - 516966
dfCNLs = DataFrame(X, CNLfeaturesStr)
insertcols!(dfCNLs, 1, ("ENTRY"=>collect((1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959))))
insertcols!(dfCNLs, 2, ("INCHIKEY1_ID"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "INCHIKEY_ID"]))
insertcols!(dfCNLs, 3, ("INCHIKEY"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "INCHIKEY"]))
insertcols!(dfCNLs, 4, ("INCHIKEYreal"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "INCHIKEYreal"]))
insertcols!(dfCNLs, 5, ("RefMatchFragRatio"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "RefMatchFragRatio"]))
insertcols!(dfCNLs, 6, ("UsrMatchFragRatio"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "UsrMatchFragRatio"]))
insertcols!(dfCNLs, 7, ("MS1Error"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "MS1Error"]))
insertcols!(dfCNLs, 8, ("MS2Error"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "MS2Error"]))
insertcols!(dfCNLs, 9, ("MS2ErrorStd"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "MS2ErrorStd"]))
insertcols!(dfCNLs, 10, ("DirectMatch"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "DirectMatch"]))
insertcols!(dfCNLs, 11, ("ReversMatch"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "ReversMatch"]))
insertcols!(dfCNLs, 12, ("FinalScoreRatio"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "FinalScoreRatio"]))
insertcols!(dfCNLs, 13, ("ISOTOPICMASS"=>inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "MS1Mass"] .- 1.007276))
dfCNLs[!, "predictRi"] = inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "predictRi"]
size(dfCNLs)  # 516966 x (13+15961+1)

desStat = describe(dfCNLs)  # 15975 x 7
desStat[14,:]

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
# 1076799 -> 1076800 rows
dfCNLsSum = dfCNLs[end:end, :]
savePath = "F:\\dfCNLsSum_8.csv"
CSV.write(savePath, dfCNLsSum)

using DataSci4Chem
massesCNLsDistrution = bar(names(dfCNLs)[14:end-1], Vector(dfCNLs[end, 14:end-1]), 
    label = false, 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    xlabel!("CNLs features")
    ylabel!("Summation")
    # Saving
    savefig(massesCNLsDistrution, "F:\\TPTNmassesCNLsDistrution_8.png")

dfCNLs = dfCNLs[1:end-1, :]
#load a model
# requires python 3.11 or 3.12
modelRF_CNL = jl.load("F:\\CocamideExtended_CNLsRi_RFwithStratification.joblib")
size(modelRF_CNL)

CNLpredictedRi = predict(modelRF_CNL, Matrix(dfCNLs[:, 13:end-1]))
dfCNLs[!, "CNLpredictRi"] = CNLpredictedRi
dfCNLs[!, "DeltaRi"] = (CNLpredictedRi - dfCNLs[:, "predictRi"]) / 1000
dfCNLs[!, "LABEL"] = inputTPTNdf[(1+516966+516966+516966+516966+516966+516966+516966):(516966+516966+516966+516966+516966+516966+516966+516959), "LABEL"]
# save, ouputing testSet df 0.3 x (3+15994+1)
savePath = "F:\\dataframeCNLsRows4TPTNModeling_8withCNLRideltaRi.csv"
CSV.write(savePath, dfCNLs)
println("done for saving csv")



outputDf = (dfCNLs[dfCNLs.LABEL .== 1, :])[:, 1:end-3]
savePath = "F:\\dataframeCNLsRows4TPTNModeling_TPOnlywithCNLRideltaRi.csv"
CSV.write(savePath, outputDf)
println("done for saving csv")

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
# 1076799 -> 1076800 rows
dfCNLsSum = outputDf[end:end, :]
savePath = "F:\\dfCNLsSum_TP.csv"
CSV.write(savePath, dfCNLsSum)


dfCNLsSum1 = CSV.read("F:\\dfCNLsSum_1.csv", DataFrame)
dfCNLsSum2 = CSV.read("F:\\dfCNLsSum_2.csv", DataFrame)
dfCNLsSum3 = CSV.read("F:\\dfCNLsSum_3.csv", DataFrame)
dfCNLsSum4 = CSV.read("F:\\dfCNLsSum_4.csv", DataFrame)
dfCNLsSum5 = CSV.read("F:\\dfCNLsSum_5.csv", DataFrame)
dfCNLsSum6 = CSV.read("F:\\dfCNLsSum_6.csv", DataFrame)
dfCNLsSum7 = CSV.read("F:\\dfCNLsSum_7.csv", DataFrame)
dfCNLsSum8 = CSV.read("F:\\dfCNLsSum_8.csv", DataFrame)
dfCNLsSum = vcat(dfCNLsSum1, dfCNLsSum2, dfCNLsSum3, dfCNLsSum4, dfCNLsSum5, dfCNLsSum6, dfCNLsSum7, dfCNLsSum8)
dfCNLsSumTP = CSV.read("F:\\dfCNLsSum_TP.csv", DataFrame)

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
# 1076799 -> 1076800 rows
dfCNLsSum = dfCNLsSum[end:end, :]
savePath = "F:\\dfCNLsSum.csv"
CSV.write(savePath, dfCNLsSum)

using DataSci4Chem
massesCNLsDistrution = bar(names(dfCNLsSum)[14:end-1], Vector(dfCNLsSum[end, 14:end-1]), 
    label = false, 
    lc = "pink", 
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    xlabel!("CNLs features")
    ylabel!("Summation")
    bar!(names(dfCNLsSumTP)[14:end-1], Vector(dfCNLsSumTP[end, 14:end-1]), 
        label = false, 
        lc = "skyblue", 
        margin = (5, :mm), 
        size = (1000,800), 
        dpi = 300)
        xlabel!("CNLs features")
        ylabel!("Summation")
    # Saving
    savefig(massesCNLsDistrution, "F:\\TPTNmassesCNLsDistrution.png")
