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

# inputing 693677 x 3+21567 df
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputTPTNdf = CSV.read("F:\\Cand_search_rr0_0612_TEST_100-400_extractedWithoutDeltaRi.csv", DataFrame)
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

# initialization for 1 more column -> 817413 x 5
inputTPTNdf[!, "CNLmasses"] .= [[]]
size(inputTPTNdf)

# NLs calculation, filtering CNL-in-interest, storing in Vector{Any}
        # filtering in CNLs features according to the pre-defined CNLs in CNLs_10mDa.csv
        # inputing 16022 candidates
        candidates_df = CSV.read("F:\\dataframe_dfTrainSetWithStratification_withCNLPredictedRi.csv", DataFrame)
        CNLfeaturesStr = names(candidates_df)[4:end-2]
        candidatesList = []
        for can in CNLfeaturesStr
            push!(candidatesList, round(parse(Float64, can), digits = 2))
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

# Reducing df size (rows)
        function getMasses(db, i, arr)
            massesArr = arr
            masses = db[i, "CNLmasses"]
            for mass in masses
                push!(massesArr, mass)
            end
            return massesArr
        end


# creating a table with 4+15961 columns features CNLs
CNLfeaturesStr
candidatesList
inputTPTNdf

# storing data in a Matrix
X = zeros(105558, 15961)

for i in 1:size(inputTPTNdf, 1)
    println(i)
    arr = []
    arr = getMasses(inputTPTNdf, i, arr)
    mumIon = round(inputTPTNdf[i, "MS1Mass"], digits = 2)
    for col in arr
        mz = findall(x->x==col, candidatesList)
        if (col <= mumIon)
            X[i, mz] .= 1
        elseif (col > mumIon)
            X[i, mz] .= -1
        end
    end
end

dfCNLs = DataFrame(X, CNLfeaturesStr)
insertcols!(dfCNLs, 1, ("ENTRY"=>collect(1:105558)))
insertcols!(dfCNLs, 2, ("INCHIKEY1_ID"=>inputTPTNdf[:, "INCHIKEY_ID"]))
insertcols!(dfCNLs, 3, ("INCHIKEY"=>inputTPTNdf[:, "INCHIKEY"]))
insertcols!(dfCNLs, 4, ("RefMatchFragRatio"=>inputTPTNdf[:, "RefMatchFragRatio"]))
insertcols!(dfCNLs, 5, ("UsrMatchFragRatio"=>inputTPTNdf[:, "UsrMatchFragRatio"]))
insertcols!(dfCNLs, 6, ("MS1Error"=>inputTPTNdf[:, "MS1Error"]))
insertcols!(dfCNLs, 7, ("MS2Error"=>inputTPTNdf[:, "MS2Error"]))
insertcols!(dfCNLs, 8, ("MS2ErrorStd"=>inputTPTNdf[:, "MS2ErrorStd"]))
insertcols!(dfCNLs, 9, ("DirectMatch"=>inputTPTNdf[:, "DirectMatch"]))
insertcols!(dfCNLs, 10, ("ReversMatch"=>inputTPTNdf[:, "ReversMatch"]))
insertcols!(dfCNLs, 11, ("FinalScoreRatio"=>inputTPTNdf[:, "FinalScoreRatio"]))
insertcols!(dfCNLs, 12, ("ISOTOPICMASS"=>inputTPTNdf[:, "MS1Mass"] .- 1.007276))
dfCNLs[!, "predictRi"] = inputTPTNdf[:, "predictRi"]
size(dfCNLs)  # 693685 x (3+1+15961)

# ouputing df 693685 x (3+1+15961)
savePath = "F:\\dataframeCNLsRows4TPTNModeling.csv"
CSV.write(savePath, dfCNLs)
println("done for saving csv")

desStat = describe(dfCNLs)  # 15965 x 7
desStat[13,:]

sumUp = []
push!(sumUp, 888888)
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, 888888)
push!(sumUp, 888888)
push!(sumUp, 888888)
push!(sumUp, 888888)
push!(sumUp, 888888)
push!(sumUp, 888888)
push!(sumUp, 888888)
push!(sumUp, 888888)
push!(sumUp, 888888)
for col in names(dfCNLs)[13:end-1]
    count = 0
    for i in 1:size(dfCNLs, 1)
        count += dfCNLs[i, col]
    end
    push!(sumUp, count)
end
push!(sumUp, 888888)
push!(dfCNLs, sumUp)
# 693685 -> 693686 rows
dfCNLs[end,:]  #693686

using DataSci4Chem
massesCNLsDistrution = bar(names(dfCNLs)[13:end-1], Vector(dfCNLs[end, 13:end-1]), 
    label = false, 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    xlabel!("CNLs features")
    ylabel!("Summation")
    # Saving
    savefig(massesCNLsDistrution, "F:\\TPTNmassesCNLsDistrution.png")

dfCNLs = dfCNLs[1:end-1, :]
#load a model
# requires python 3.11 or 3.12
modelRF_CNL = jl.load("F:\\CocamideExtended_CNLsRi_RFwithStratification.joblib")
size(modelRF_CNL)

CNLpredictedRi = predict(modelRF_CNL, Matrix(dfCNLs[:, 12:end-1]))
dfCNLs[!, "CNLpredictRi"] = CNLpredictedRi
dfCNLs[!, "DeltaRi"] = (CNLpredictedRi - dfCNLs[:, "predictRi"]) / 1000
dfCNLs[!, "LABEL"] = inputTPTNdf[:, "LABEL"]
# save, ouputing testSet df 0.3 x (3+15994+1)
savePath = "F:\\Cand_search_rr0_0612_TEST_100-400_extractedWithDeltaRi.csv"
CSV.write(savePath, dfCNLs)