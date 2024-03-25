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

# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage, withDeltaRipredictTPTN, p0, p1
# inputing 3283078 x 4+8+1+2+1+1+2+1+2 df
inputDB_trainWithDeltaRi = CSV.read("F:\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_trainWithDeltaRi, :pTP_train1 => :p0)
rename!(inputDB_trainWithDeltaRi, :pTP_train2 => :p1)

inputDB_trainWithoutDeltaRi = CSV.read("F:\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_trainWithoutDeltaRi, :pTP_train1 => :p0)
rename!(inputDB_trainWithoutDeltaRi, :pTP_train2 => :p1)

# inputing 820770 x 4+8+1+2+1+1+2+1+2 df
inputDB_testWithDeltaRi = CSV.read("F:\\dataframeTPTNModeling_TestDF_withDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_testWithDeltaRi, :pTP_test1 => :p0)
rename!(inputDB_testWithDeltaRi, :pTP_test2 => :p1)

inputDB_testWithoutDeltaRi = CSV.read("F:\\dataframeTPTNModeling_TestDF_withoutDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_testWithoutDeltaRi, :pTP_test1 => :p0)
rename!(inputDB_testWithoutDeltaRi, :pTP_test2 => :p1)

describe(inputDB_trainWithDeltaRi)[end-5:end, :]

# concate into 4103848 x 22 df
inputDB_WithDeltaRi = vcat(inputDB_trainWithDeltaRi, inputDB_testWithDeltaRi)
inputDB_WithoutDeltaRi = vcat(inputDB_trainWithoutDeltaRi, inputDB_testWithoutDeltaRi)

# --------------------------------------------------------------------------------------------------
# inputing 4103848 x 4+8+1+2+1+1+2+1+2 df
inputDB_WholeWithDeltaRi = CSV.read("F:\\dataframeTPTNModeling_WholeDF_withDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_WholeWithDeltaRi, :pTP_train1 => :p0)
rename!(inputDB_WholeWithDeltaRi, :pTP_train2 => :p1)

inputDB_WholeWithoutDeltaRi = CSV.read("F:\\dataframeTPTNModeling_WholeDF_withoutDeltaRiandPredictedTPTNandpTP.csv", DataFrame)
rename!(inputDB_WholeWithoutDeltaRi, :pTP_train1 => :p0)
rename!(inputDB_WholeWithoutDeltaRi, :pTP_train2 => :p1)

# ==================================================================================================
# prepare plotting confusion matrix
inputDB_WithDeltaRi[!, "CM"] .= String("")
inputDB_WithDeltaRi_TP = 0  # 218195
inputDB_WithDeltaRi_FP = 0  # 13459
inputDB_WithDeltaRi_TN = 0  # 3864817
inputDB_WithDeltaRi_FN = 0  # 7377
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

# save, ouputing df 4103848 x 22+1 df 
savePath = "F:\\dataframePostPredict_withDeltaRi.csv"
CSV.write(savePath, inputDB_WithDeltaRi)

# --------------------------------------------------------------------------------------------------
inputDB_WithoutDeltaRi[!, "CM"] .= String("")
inputDB_WithoutDeltaRi_TP = 0  # 209353
inputDB_WithoutDeltaRi_FP = 0  # 221313
inputDB_WithoutDeltaRi_TN = 0  # 3656963
inputDB_WithoutDeltaRi_FN = 0  # 16219
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

# save, ouputing df 4103848 x 22+1 df 
savePath = "F:\\dataframePostPredict_withoutDeltaRi.csv"
CSV.write(savePath, inputDB_WithoutDeltaRi)

# --------------------------------------------------------------------------------------------------
inputDB_WholeWithDeltaRi[!, "CM"] .= String("")
inputDB_WholeWithDeltaRi_TP = 0  # 225448
inputDB_WholeWithDeltaRi_FP = 0  # 8813
inputDB_WholeWithDeltaRi_TN = 0  # 3869463
inputDB_WholeWithDeltaRi_FN = 0  # 124
for i in 1:size(inputDB_WholeWithDeltaRi , 1)
    if (inputDB_WholeWithDeltaRi[i, "LABEL"] == 1 && inputDB_WholeWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_WholeWithDeltaRi[i, "CM"] = "TP"
        inputDB_WholeWithDeltaRi_TP += 1
    elseif (inputDB_WholeWithDeltaRi[i, "LABEL"] == 0 && inputDB_WholeWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_WholeWithDeltaRi[i, "CM"] = "FP"
        inputDB_WholeWithDeltaRi_FP += 1
    elseif (inputDB_WholeWithDeltaRi[i, "LABEL"] == 0 && inputDB_WholeWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
        inputDB_WholeWithDeltaRi[i, "CM"] = "TN"
        inputDB_WholeWithDeltaRi_TN += 1
    elseif (inputDB_WholeWithDeltaRi[i, "LABEL"] == 1 && inputDB_WholeWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
        inputDB_WholeWithDeltaRi[i, "CM"] = "FN"
        inputDB_WholeWithDeltaRi_FN += 1
    end
end
describe(inputDB_WholeWithDeltaRi)[end-5:end, :]

CM_WholeWith = zeros(2, 2)
CM_WholeWith[2, 1] = inputDB_WholeWithDeltaRi_TP
CM_WholeWith[2, 2] = inputDB_WholeWithDeltaRi_FP
CM_WholeWith[1, 2] = inputDB_WholeWithDeltaRi_TN
CM_WholeWith[1, 1] = inputDB_WholeWithDeltaRi_FN

# save, ouputing df 4103848 x 22+1 df 
savePath = "F:\\dataframePostPredict_WholeWithDeltaRi.csv"
CSV.write(savePath, inputDB_WholeWithDeltaRi)

# --------------------------------------------------------------------------------------------------
inputDB_WholeWithoutDeltaRi[!, "CM"] .= String("")
inputDB_WholeWithoutDeltaRi_TP = 0  # 225445
inputDB_WholeWithoutDeltaRi_FP = 0  # 212224
inputDB_WholeWithoutDeltaRi_TN = 0  # 3666052
inputDB_WholeWithoutDeltaRi_FN = 0  # 127
for i in 1:size(inputDB_WholeWithoutDeltaRi , 1)
    if (inputDB_WholeWithoutDeltaRi[i, "LABEL"] == 1 && inputDB_WholeWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
        inputDB_WholeWithoutDeltaRi[i, "CM"] = "TP"
        inputDB_WholeWithoutDeltaRi_TP += 1
    elseif (inputDB_WholeWithoutDeltaRi[i, "LABEL"] == 0 && inputDB_WholeWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
        inputDB_WholeWithoutDeltaRi[i, "CM"] = "FP"
        inputDB_WholeWithoutDeltaRi_FP += 1
    elseif (inputDB_WholeWithoutDeltaRi[i, "LABEL"] == 0 && inputDB_WholeWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
        inputDB_WholeWithoutDeltaRi[i, "CM"] = "TN"
        inputDB_WholeWithoutDeltaRi_TN += 1
    elseif (inputDB_WholeWithoutDeltaRi[i, "LABEL"] == 1 && inputDB_WholeWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
        inputDB_WholeWithoutDeltaRi[i, "CM"] = "FN"
        inputDB_WholeWithoutDeltaRi_FN += 1
    end
end
describe(inputDB_WholeWithoutDeltaRi)[end-5:end, :]

CM_WholeWithout = zeros(2, 2)
CM_WholeWithout[2, 1] = inputDB_WholeWithoutDeltaRi_TP
CM_WholeWithout[2, 2] = inputDB_WholeWithoutDeltaRi_FP
CM_WholeWithout[1, 2] = inputDB_WholeWithoutDeltaRi_TN
CM_WholeWithout[1, 1] = inputDB_WholeWithoutDeltaRi_FN

# save, ouputing df 4103848 x 22+1 df 
savePath = "F:\\dataframePostPredict_WholeWithoutDeltaRi.csv"
CSV.write(savePath, inputDB_WholeWithoutDeltaRi)

# ==================================================================================================
# plot confusion matrix
#= layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
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
        annotate!(["1"], ["1"], ["TP\n218195"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n13459"], subplot = 1, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n7377"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n3864817"], subplot = 1)
heatmap!(["1", "0"], ["0", "1"], CM_without, cmap = :viridis, cbar = :true, 
        clims = (10000, 400000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", ylabel = "Predicted", 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n209353"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n221313"], subplot = 2)
        annotate!(["1"], ["0"], ["FN\n16219"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n3656963"], subplot = 2)
savefig(outplotCM, "F:\\TPTNPrediction_RFallCM.png") =#

# --------------------------------------------------------------------------------------------------
layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
default(grid = false, legend = false)
gr()
wholeOutplotCM = plot(layout = layout, link = :both, 
        size = (1400, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_WholeWith, cmap = :viridis, cbar = :none, 
        clims = (10000, 400000), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", ylabel = "Predicted", 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n225448"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n8813"], subplot = 1, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n124"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n3869463"], subplot = 1)
heatmap!(["1", "0"], ["0", "1"], CM_WholeWithout, cmap = :viridis, cbar = :true, 
        clims = (10000, 400000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", ylabel = "Predicted", 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n225445"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n212224"], subplot = 2)
        annotate!(["1"], ["0"], ["FN\n127"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n3666052"], subplot = 2)
savefig(wholeOutplotCM, "F:\\TPTNPrediction_RFwholeCM.png")

# ==================================================================================================

# prepare plotting P(TP)threshold-to-TPR curve
# 4103848 x 22+1 df
inputDB_WithDeltaRi = CSV.read("F:\\dataframePostPredict_withDeltaRi.csv", DataFrame)
sort!(inputDB_WithDeltaRi, [:p1], rev = true)
for i in 1:size(inputDB_WithDeltaRi, 1)
    inputDB_WithDeltaRi[i, "p1"] = round(float(inputDB_WithDeltaRi[i, "p1"]), digits = 2)
end

inputDB_WholeWithDeltaRi = CSV.read("F:\\dataframePostPredict_WholeWithDeltaRi.csv", DataFrame)
sort!(inputDB_WholeWithDeltaRi, [:p1], rev = true)
for i in 1:size(inputDB_WholeWithDeltaRi, 1)
    inputDB_WholeWithDeltaRi[i, "p1"] = round(float(inputDB_WholeWithDeltaRi[i, "p1"]), digits = 2)
end


function get1rate(df, thd)
    TP = 0  # 
    FN = 0  # 
    TN = 0  # 
    FP = 0  # 
    for i in 1:size(df , 1)
        if (df[i, "LABEL"] == 1 && df[i, "p1"] >= thd)
            TP += 1
        elseif (df[i, "LABEL"] == 1 && df[i, "p1"] < thd)
            FN += 1
        elseif (df[i, "LABEL"] == 0 && df[i, "p1"] >= thd)
            FP += 1
        elseif (df[i, "LABEL"] == 0 && df[i, "p1"] < thd)
            TN += 1
        end
    end
    return (TP / (TP + FN)), (FN / (TP + FN)), (FP / (FP + TP))
end


withDeltaRi_TPR = []
withDeltaRi_FNR = []
withDeltaRi_FDR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
for temp in Array(inputDB_WithDeltaRi[:, "p1"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR, FDR = get1rate(inputDB_WithDeltaRi, prob)
        push!(withDeltaRi_TPR, TPR)
        push!(withDeltaRi_FNR, FNR)
        push!(withDeltaRi_FDR, FDR)
    else
        push!(withDeltaRi_TPR, TPR)
        push!(withDeltaRi_FNR, FNR)
        push!(withDeltaRi_FDR, FDR)
    end
end

inputDB_WithDeltaRi[!, "TPR"] = withDeltaRi_TPR
inputDB_WithDeltaRi[!, "FNR"] = withDeltaRi_FNR
inputDB_WithDeltaRi[!, "FDR"] = withDeltaRi_FDR

# save, ouputing df 4103848 x 23+3 df 
savePath = "F:\\dataframePostPredict_TPRFNRFDR.csv"
CSV.write(savePath, inputDB_WithDeltaRi)

# --------------------------------------------------------------------------------------------------
wholeWithDeltaRi_TPR = []
wholeWithDeltaRi_FNR = []
wholeWithDeltaRi_FDR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
for temp in Array(inputDB_WholeWithDeltaRi[:, "p1"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR, FDR = get1rate(inputDB_WholeWithDeltaRi, prob)
        push!(wholeWithDeltaRi_TPR, TPR)
        push!(wholeWithDeltaRi_FNR, FNR)
        push!(wholeWithDeltaRi_FDR, FDR)
    else
        push!(wholeWithDeltaRi_TPR, TPR)
        push!(wholeWithDeltaRi_FNR, FNR)
        push!(wholeWithDeltaRi_FDR, FDR)
    end
end

inputDB_WholeWithDeltaRi[!, "TPR"] = wholeWithDeltaRi_TPR
inputDB_WholeWithDeltaRi[!, "FNR"] = wholeWithDeltaRi_FNR
inputDB_WholeWithDeltaRi[!, "FDR"] = wholeWithDeltaRi_FDR

# save, ouputing df 4103848 x 23+3 df 
savePath = "F:\\dataframePostPredict_TPRFNRFDR.csv"
CSV.write(savePath, inputDB_WholeWithDeltaRi)

# ==================================================================================================
# plot P(1)threshold-to-TPR & P(1)threshold-to-TNR
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()

describe(inputDB_WithDeltaRi)[end-4:end, :]

outplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)

plot!(inputDB_WithDeltaRi[:, end-4], [inputDB_WithDeltaRi[:, end-2] inputDB_WithDeltaRi[:, end-1]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) threshold", 
        label = ["True positive rate" "False negative rate"], 
        legend = :left, 
        size = (1200,600), 
        dpi = 300)

plot!(inputDB_WithDeltaRi[:, end], inputDB_WithDeltaRi[:, end-2], 
        subplot = 2, framestyle = :box, 
        xlabel = "False discovery rate", 
        ylabel = "True positive rate", 
        legend = false, 
        size = (1200,600), 
        dpi = 300)

savefig(outplotP1toRate, "F:\\TPTNPrediction_P1threshold2TPRFNRFDR.png")
