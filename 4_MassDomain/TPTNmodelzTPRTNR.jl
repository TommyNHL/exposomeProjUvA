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
#using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots
#using ProgressBars
#using PyPlot
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#using PyCall, Conda                 #using python packages
#pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models
f1_score = pyimport("sklearn.metrics").f1_score
matthews_corrcoef = pyimport("sklearn.metrics").matthews_corrcoef

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
@sk_import metrics: recall_score
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

# columns: ENTRY, INCHIKEY, ISOTOPICMASS, CNLs, predictRi
# inputing 3308576 x 4+9+2+1+2+1+2 df
inputDB_trainWithDeltaRi = CSV.read("F:\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_trainWithDeltaRi, :pTP_train1 => :p0)
rename!(inputDB_trainWithDeltaRi, :pTP_train2 => :p1)

inputDB_trainWithoutDeltaRi = CSV.read("F:\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_trainWithoutDeltaRi, :pTP_train1 => :p0)
rename!(inputDB_trainWithoutDeltaRi, :pTP_train2 => :p1)

# inputing 827145 x 4+9+2+1+2+1+2 df
inputDB_testWithDeltaRi = CSV.read("F:\\dataframeTPTNModeling_TestDF_withDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_testWithDeltaRi, :pTP_test1 => :p0)
rename!(inputDB_testWithDeltaRi, :pTP_test2 => :p1)

inputDB_testWithoutDeltaRi = CSV.read("F:\\dataframeTPTNModeling_TestDF_withoutDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_testWithoutDeltaRi, :pTP_test1 => :p0)
rename!(inputDB_testWithoutDeltaRi, :pTP_test2 => :p1)

describe(inputDB_trainWithDeltaRi)[end-5:end, :]

# concate into 4135721 x 21 df
inputDB_WithDeltaRi = vcat(inputDB_trainWithDeltaRi, inputDB_testWithDeltaRi)
inputDB_WithoutDeltaRi = vcat(inputDB_trainWithoutDeltaRi, inputDB_testWithoutDeltaRi)

# prepare plotting confusion matrix
inputDB_WithDeltaRi[!, "CM"] .= String("")
inputDB_WithDeltaRi_TP = 0  # 221830
inputDB_WithDeltaRi_FP = 0  # 13721
inputDB_WithDeltaRi_TN = 0  # 3892771
inputDB_WithDeltaRi_FN = 0  # 7399
for i in 1:size(inputDB_WithDeltaRi , 1)
    if (inputDB_WithDeltaRi[i, "LABEL"] == 1 && inputDB_WithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_WithDeltaRi[i, "CM"] = "TP"
        inputDB_WithDeltaRi_TP += 1
    elseif (inputDB_WithDeltaRi[i, "LABEL"] == 0 && inputDB_WithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_WithDeltaRi[i, "CM"] = "FP"
        inputDB_WithDeltaRi_FP += 1
    elseif (inputDB_WithDeltaRi[i, "LABEL"] == 0 && inputDB_WithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
        inputDB_WithDeltaRi[i, "CM"] = "TN"
        inputDB_WithDeltaRi_TN += 1
    elseif (inputDB_WithDeltaRi[i, "LABEL"] == 1 && inputDB_WithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
        inputDB_WithDeltaRi[i, "CM"] = "FN"
        inputDB_WithDeltaRi_FN += 1
    end
end
describe(inputDB_WithDeltaRi)[end-5:end, :]

CM_with = zeros(2, 2)
CM_with[2, 1] = inputDB_WithDeltaRi_TP
CM_with[2, 2] = inputDB_WithDeltaRi_FP
CM_with[1, 2] = inputDB_WithDeltaRi_TN
CM_with[1, 1] = inputDB_WithDeltaRi_FN

# save, ouputing df 4135721 x 22 df 
savePath = "F:\\dataframePostPredict_withDeltaRi.csv"
CSV.write(savePath, inputDB_WithDeltaRi)

# --------------------------------------------------------------------------------------------------
inputDB_WithoutDeltaRi[!, "CM"] .= String("")
inputDB_WithoutDeltaRi_TP = 0  # 213044
inputDB_WithoutDeltaRi_FP = 0  # 226382
inputDB_WithoutDeltaRi_TN = 0  # 3680110
inputDB_WithoutDeltaRi_FN = 0  # 16185
for i in 1:size(inputDB_WithoutDeltaRi , 1)
    if (inputDB_WithoutDeltaRi[i, "LABEL"] == 1 && inputDB_WithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
        inputDB_WithoutDeltaRi[i, "CM"] = "TP"
        inputDB_WithoutDeltaRi_TP += 1
    elseif (inputDB_WithoutDeltaRi[i, "LABEL"] == 0 && inputDB_WithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
      inputDB_WithoutDeltaRi[i, "CM"] = "FP"
        inputDB_WithoutDeltaRi_FP += 1
    elseif (inputDB_WithoutDeltaRi[i, "LABEL"] == 0 && inputDB_WithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
      inputDB_WithoutDeltaRi[i, "CM"] = "TN"
        inputDB_WithoutDeltaRi_TN += 1
    elseif (inputDB_WithoutDeltaRi[i, "LABEL"] == 1 && inputDB_WithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
      inputDB_WithoutDeltaRi[i, "CM"] = "FN"
        inputDB_WithoutDeltaRi_FN += 1
    end
end
describe(inputDB_WithoutDeltaRi)[end-5:end, :]

CM_without = zeros(2, 2)
CM_without[2, 1] = inputDB_WithoutDeltaRi_TP
CM_without[2, 2] = inputDB_WithoutDeltaRi_FP
CM_without[1, 2] = inputDB_WithoutDeltaRi_TN
CM_without[1, 1] = inputDB_WithoutDeltaRi_FN

# save, ouputing df 4135721 x 22 df 
savePath = "F:\\dataframePostPredict_withoutDeltaRi.csv"
CSV.write(savePath, inputDB_WithoutDeltaRi)

# plot confusion matrix
layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
default(grid = false, legend = false)
gr()
outplotCM = plot(layout = layout, link = :both, 
        size = (1400, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_with, cmap = :viridis, cbar = :none, 
        clims = (10000, 400000), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", ylabel = "Predicted", 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n221830"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n13721"], subplot = 1)
        annotate!(["1"], ["0"], ["FN\n7399"], subplot = 1)
        annotate!(["0"], ["0"], ["TN\n3892770"], subplot = 1)
heatmap!(["1", "0"], ["0", "1"], CM_without, cmap = :viridis, cbar = :true, 
        clims = (10000, 400000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", ylabel = "Predicted", 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n213044"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n226382"], subplot = 2)
        annotate!(["1"], ["0"], ["FN\n16185"], subplot = 2)
        annotate!(["0"], ["0"], ["TN\n3680110"], subplot = 2)
        savefig(outplotCM, "F:\\TPTNPrediction_RFallCM.png")

# ==================================================================================================
# prepare plotting P(TP)threshold-to-TPR curve
inputDB_WithDeltaRi = CSV.read("F:\\dataframePostPredict_withDeltaRi.csv", DataFrame)
inputDB_WithDeltaRi_1 = inputDB_WithDeltaRi[inputDB_WithDeltaRi.LABEL .== 1, :]
sort!(inputDB_WithDeltaRi_1, [:p1], rev = true)
for i in 1:size(inputDB_WithDeltaRi_1, 1)
    inputDB_WithDeltaRi_1[i, "p1"] = round(float(inputDB_WithDeltaRi_1[i, "p1"]), digits = 2)
end

inputDB_WithoutDeltaRi = CSV.read("F:\\dataframePostPredict_withoutDeltaRi.csv", DataFrame)
inputDB_WithoutDeltaRi_1 = inputDB_WithoutDeltaRi[inputDB_WithoutDeltaRi.LABEL .== 1, :]
sort!(inputDB_WithoutDeltaRi_1, [:p1], rev = true)
for i in 1:size(inputDB_WithoutDeltaRi_1, 1)
    inputDB_WithoutDeltaRi_1[i, "p1"] = round(float(inputDB_WithoutDeltaRi_1[i, "p1"]), digits = 2)
end

describe(inputDB_WithDeltaRi_1)[end-5:end, :]
inputDB_WithDeltaRi_1[210229, "p1"]

function get1TPRnFNR(df, thd)
    TP = 0  # 
    FN = 0  # 
    for i in 1:size(df , 1)
        if (df[i, "p1"] >= thd)
            TP += 1
        elseif (df[i, "p1"] < thd)
            FN += 1
        end
    end
    return (TP / (TP + FN)), (FN / (TP + FN))
end


withDeltaRi_TPR = []
withDeltaRi_FNR = []
prob = -1
TPR = 0
FNR = 0
for temp in Array(inputDB_WithDeltaRi_1[:, "p1"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR = get1TPRnFNR(inputDB_WithDeltaRi_1, prob)
        push!(withDeltaRi_TPR, TPR)
        push!(withDeltaRi_FNR, FNR)
    else
        push!(withDeltaRi_TPR, TPR)
        push!(withDeltaRi_FNR, FNR)
    end
end

withoutDeltaRi_TPR = []
withoutDeltaRi_FNR = []
prob = -1
TPR = 0
FNR = 0
for temp in Array(inputDB_WithoutDeltaRi_1[:, "p1"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR = get1TPRnFNR(inputDB_WithoutDeltaRi_1, prob)
        push!(withoutDeltaRi_TPR, TPR)
        push!(withoutDeltaRi_FNR, FNR)
    else
        push!(withoutDeltaRi_TPR, TPR)
        push!(withoutDeltaRi_FNR, FNR)
    end
end

inputDB_WithDeltaRi_1[!, "TPR"] = withDeltaRi_TPR
inputDB_WithDeltaRi_1[!, "FNR"] = withDeltaRi_FNR
inputDB_WithoutDeltaRi_1[!, "TPR"] = withoutDeltaRi_TPR
inputDB_WithoutDeltaRi_1[!, "FNR"] = withoutDeltaRi_FNR

# save, ouputing df 4135721 x 22 df 
savePath = "F:\\dataframePostPredict_withDeltaRiTPRFNR.csv"
CSV.write(savePath, inputDB_WithDeltaRi_1)
# save, ouputing df 4135721 x 22 df 
savePath2 = "F:\\dataframePostPredict_withoutDeltaRiTPRFNR.csv"
CSV.write(savePath2, inputDB_WithoutDeltaRi_1)
# --------------------------------------------------------------------------------------------------
# prepare plotting P(TP)threshold-to-TPR curve
inputDB_WithDeltaRi_0 = inputDB_WithDeltaRi[inputDB_WithDeltaRi.LABEL .== 0, :]
sort!(inputDB_WithDeltaRi_0, [:p1], rev = true)
for i in 1:size(inputDB_WithDeltaRi_0, 1)
    inputDB_WithDeltaRi_0[i, "p1"] = round(float(inputDB_WithDeltaRi_0[i, "p1"]), digits = 2)
end

inputDB_WithoutDeltaRi_0 = inputDB_WithoutDeltaRi[inputDB_WithoutDeltaRi.LABEL .== 0, :]
sort!(inputDB_WithoutDeltaRi_0, [:p1], rev = true)
for i in 1:size(inputDB_WithoutDeltaRi_0, 1)
    inputDB_WithoutDeltaRi_0[i, "p1"] = round(float(inputDB_WithoutDeltaRi_0[i, "p1"]), digits = 2)
end

describe(inputDB_WithDeltaRi_0)[end-5:end, :]
inputDB_WithDeltaRi_0[210229, "p1"]

function get1TNRFPR(df, thd)
    TN = 0  # 
    FP = 0  # 
    for i in 1:size(df , 1)
        if (df[i, "p1"] <= thd)
            TN += 1
        elseif (df[i, "p1"] > thd)
            FP += 1
        end
    end
    return (TN / (TN + FP)), (FP / (TN + FP))
end

withDeltaRi_TNR = []
withDeltaRi_FPR = []
prob = -1
TNR = 0
FPR = 0
for temp in Array(inputDB_WithDeltaRi_0[:, "p1"])
    if (temp != prob)
        println(temp)
        prob = temp
        TNR, FPR = get1TNRFPR(inputDB_WithDeltaRi_0, prob)
        push!(withDeltaRi_TNR, TNR)
        push!(withDeltaRi_FPR, FPR)
    else
        push!(withDeltaRi_TNR, TNR)
        push!(withDeltaRi_FPR, FPR)
    end
end

withoutDeltaRi_TNR = []
withoutDeltaRi_FPR = []
prob = -1
TNR = 0
FPR = 0
for temp in Array(inputDB_WithoutDeltaRi_0[:, "p1"])
    if (temp != prob)
        println(temp)
        prob = temp
        TNR, FPR = get1TNRFPR(inputDB_WithoutDeltaRi_0, prob)
        push!(withoutDeltaRi_TNR, TNR)
        push!(withoutDeltaRi_FPR, FPR)
    else
        push!(withoutDeltaRi_TNR, TNR)
        push!(withoutDeltaRi_FPR, FPR)
    end
end

inputDB_WithDeltaRi_0[!, "TNR"] = withDeltaRi_TNR
inputDB_WithDeltaRi_0[!, "FPR"] = withDeltaRi_FPR
inputDB_WithoutDeltaRi_0[!, "TNR"] = withoutDeltaRi_TNR
inputDB_WithoutDeltaRi_0[!, "FPR"] = withoutDeltaRi_FPR

# save, ouputing df 4135721 x 22 df 
savePath3 = "F:\\dataframePostPredict_withDeltaRiTNRFPR.csv"
CSV.write(savePath3, inputDB_WithDeltaRi_0)
# save, ouputing df 4135721 x 22 df 
savePath4 = "F:\\dataframePostPredict_withoutDeltaRiTNRFPR.csv"
CSV.write(savePath4, inputDB_WithoutDeltaRi_0)

# ==================================================================================================
# plot P(1)threshold-to-TPR & P(1)threshold-to-TNR
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()

outplotP12TR = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)

plot!(inputDB_WithDeltaRi_1[:, end-3], [inputDB_WithDeltaRi_1[:, end-1] inputDB_WithDeltaRi_1[:, end]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) threshold", 
        label = ["True positive rate" "False negative rate"], 
        legend = :left, 
        size = (1200,600), 
        dpi = 300)

plot!(inputDB_WithDeltaRi_0[:, end-3], [inputDB_WithDeltaRi_0[:, end-1] inputDB_WithDeltaRi_0[:, end]], 
        subplot = 2, framestyle = :box, 
        xlabel = "P(1) threshold", 
        label = ["True negative rate" "False positive rate"], 
        legend = :left, 
        size = (1200,600), 
        dpi = 300)

savefig(outplotP12TR, "F:\\TPTNPrediction_P1threshold2TPRTNR.png")
