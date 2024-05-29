VERSION
using Pkg
#Pkg.add("ScikitLearn")
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

## import packages ##
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
# --------------------------------------------------------------------------------------------------
# inputing 4103848 x 4+8+1+2+1+1+2+1+2 df
inputDB_WholeWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv", DataFrame)
inputDB_WholeWithoutDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv", DataFrame)
inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv", DataFrame)
inputDB_PestWithoutDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv", DataFrame)

inputDB_WholeWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE.csv", DataFrame)
inputDB_WholeWithoutDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE.csv", DataFrame)
inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE.csv", DataFrame)
inputDB_PestWithoutDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE.csv", DataFrame)
# ==================================================================================================
# prepare plotting confusion matrix
# --------------------------------------------------------------------------------------------------
inputDB_WholeWithDeltaRi[!, "CM"] .= String("")
inputDB_WholeWithDeltaRi_TP = 0
inputDB_WholeWithDeltaRi_FP = 0
inputDB_WholeWithDeltaRi_TN = 0
inputDB_WholeWithDeltaRi_FN = 0
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
#savePath = "F:\\UvA\\dataframePostPredict_WholeWithDeltaRi_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframePostPredict_WholeWithDeltaRi_0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB_WholeWithDeltaRi)

# --------------------------------------------------------------------------------------------------
inputDB_WholeWithoutDeltaRi[!, "CM"] .= String("")
inputDB_WholeWithoutDeltaRi_TP = 0
inputDB_WholeWithoutDeltaRi_FP = 0
inputDB_WholeWithoutDeltaRi_TN = 0
inputDB_WholeWithoutDeltaRi_FN = 0
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
#savePath = "F:\\UvA\\dataframePostPredict_WholeWithoutDeltaRi_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframePostPredict_WholeWithoutDeltaRi_0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB_WholeWithoutDeltaRi)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
inputDB_PestWithDeltaRi[!, "CM"] .= String("")
inputDB_PestWithDeltaRi_TP = 0
inputDB_PestWithDeltaRi_FP = 0
inputDB_PestWithDeltaRi_TN = 0
inputDB_PestWithDeltaRi_FN = 0
for i in 1:size(inputDB_PestWithDeltaRi , 1)
    if (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_PestWithDeltaRi[i, "CM"] = "TP"
        inputDB_PestWithDeltaRi_TP += 1
    elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_PestWithDeltaRi[i, "CM"] = "FP"
        inputDB_PestWithDeltaRi_FP += 1
    elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
        inputDB_PestWithDeltaRi[i, "CM"] = "TN"
        inputDB_PestWithDeltaRi_TN += 1
    elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
        inputDB_PestWithDeltaRi[i, "CM"] = "FN"
        inputDB_PestWithDeltaRi_FN += 1
    end
end
describe(inputDB_PestWithDeltaRi)[end-5:end, :]

CM_PestWith = zeros(2, 2)
CM_PestWith[2, 1] = inputDB_PestWithDeltaRi_TP
CM_PestWith[2, 2] = inputDB_PestWithDeltaRi_FP
CM_PestWith[1, 2] = inputDB_PestWithDeltaRi_TN
CM_PestWith[1, 1] = inputDB_PestWithDeltaRi_FN

# save, ouputing df 4103848 x 22+1 df 
#savePath = "F:\\UvA\\dataframePostPredict_PestWithDeltaRi_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframePostPredict_PestWithDeltaRi_0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)

# --------------------------------------------------------------------------------------------------
inputDB_PestWithoutDeltaRi[!, "CM"] .= String("")
inputDB_PestWithoutDeltaRi_TP = 0
inputDB_PestWithoutDeltaRi_FP = 0
inputDB_PestWithoutDeltaRi_TN = 0
inputDB_PestWithoutDeltaRi_FN = 0
for i in 1:size(inputDB_PestWithoutDeltaRi , 1)
    if (inputDB_PestWithoutDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
        inputDB_PestWithoutDeltaRi[i, "CM"] = "TP"
        inputDB_PestWithoutDeltaRi_TP += 1
    elseif (inputDB_PestWithoutDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
        inputDB_PestWithoutDeltaRi[i, "CM"] = "FP"
        inputDB_PestWithoutDeltaRi_FP += 1
    elseif (inputDB_PestWithoutDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
        inputDB_PestWithoutDeltaRi[i, "CM"] = "TN"
        inputDB_PestWithoutDeltaRi_TN += 1
    elseif (inputDB_PestWithoutDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
        inputDB_PestWithoutDeltaRi[i, "CM"] = "FN"
        inputDB_PestWithoutDeltaRi_FN += 1
    end
end
describe(inputDB_PestWithoutDeltaRi)[end-5:end, :]

CM_PestWithout = zeros(2, 2)
CM_PestWithout[2, 1] = inputDB_PestWithoutDeltaRi_TP
CM_PestWithout[2, 2] = inputDB_PestWithoutDeltaRi_FP
CM_PestWithout[1, 2] = inputDB_PestWithoutDeltaRi_TN
CM_PestWithout[1, 1] = inputDB_PestWithoutDeltaRi_FN

# save, ouputing df 4103848 x 22+1 df 
#savePath = "F:\\UvA\\dataframePostPredict_PestWithoutDeltaRi_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframePostPredict_PestWithoutDeltaRi_0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB_PestWithoutDeltaRi)

# ==================================================================================================
# ==================================================================================================
# plot confusion matrix
layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
default(grid = false, legend = false)
gr()
wholeOutplotCM = plot(layout = layout, link = :both, 
        size = (1400, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_WholeWith, cmap = :viridis, cbar = :true, 
        clims = (5000, 200000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n171,071"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n191,259"], subplot = 2)
        annotate!(["1"], ["0"], ["FN\n9,436"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n1,674,598"], subplot = 2)
heatmap!(["1", "0"], ["0", "1"], CM_WholeWithout, cmap = :viridis, cbar = :none, 
        clims = (5000, 200000), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n167,549"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n231,534"], subplot = 1)
        annotate!(["1"], ["0"], ["FN\n12,958"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n1,634,323"], subplot = 1)
#savefig(wholeOutplotCM, "F:\\UvA\\TPTNPrediction_RFwholeCM_0d5FinalScoreRatio.png")
savefig(wholeOutplotCM, "F:\\UvA\\TPTNPrediction_RFwholeCM_0d5FinalScoreRatioDE.png")

# --------------------------------------------------------------------------------------------------
layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
default(grid = false, legend = false)
gr()
wholeOutplotCM = plot(layout = layout, link = :both, 
        size = (1400, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_PestWith, cmap = :viridis, cbar = :true, 
        clims = (250, 10000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n7,089"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n1,985"], subplot = 2, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n1,757"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n51,177"], subplot = 2)
heatmap!(["1", "0"], ["0", "1"], CM_PestWithout, cmap = :viridis, cbar = :none, 
        clims = (250, 10000), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n7,152"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n2,109"], subplot = 1, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n1,694"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n51,053"], subplot = 1)
#savefig(wholeOutplotCM, "F:\\UvA\\TPTNPrediction_RFpestCM_0d5FinalScoreRatio.png")
savefig(wholeOutplotCM, "F:\\UvA\\TPTNPrediction_RFpestCM_0d5FinalScoreRatioDE.png")

# ==================================================================================================

# prepare plotting P(TP)threshold-to-TPR curve
# 4103848 x 22+1 df
#inputDB_WholeWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_WholeWithDeltaRi_0d5FinalScoreRatio.csv", DataFrame)
inputDB_WholeWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_WholeWithDeltaRi_0d5FinalScoreRatioDE.csv", DataFrame)
sort!(inputDB_WholeWithDeltaRi, [:"p(1)"], rev = true)
for i in 1:size(inputDB_WholeWithDeltaRi, 1)
    inputDB_WholeWithDeltaRi[i, "p(1)"] = round(float(inputDB_WholeWithDeltaRi[i, "p(1)"]), digits = 2)
end

#inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_PestWithDeltaRi_0d5FinalScoreRatio.csv", DataFrame)
inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_PestWithDeltaRi_0d5FinalScoreRatioDE.csv", DataFrame)
sort!(inputDB_PestWithDeltaRi, [:"p(1)"], rev = true)
for i in 1:size(inputDB_PestWithDeltaRi, 1)
    inputDB_PestWithDeltaRi[i, "p(1)"] = round(float(inputDB_PestWithDeltaRi[i, "p(1)"]), digits = 2)
end

function get1rate(df, thd)
    TP = 0  # 
    FN = 0  # 
    TN = 0  # 
    FP = 0  # 
    for i in 1:size(df , 1)
        if (df[i, "LABEL"] == 1 && df[i, "p(1)"] >= thd)
            TP += 1
        elseif (df[i, "LABEL"] == 1 && df[i, "p(1)"] < thd)
            FN += 1
        elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] >= thd)
            FP += 1
        elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] < thd)
            TN += 1
        end
    end
    return (TP / (TP + FN)), (FN / (TP + FN)), (FP / (FP + TP))
end

# --------------------------------------------------------------------------------------------------
wholeWithDeltaRi_TPR = []
wholeWithDeltaRi_FNR = []
wholeWithDeltaRi_FDR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
for temp in Array(inputDB_WholeWithDeltaRi[:, "p(1)"])
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
#savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_Whole0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_Whole0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB_WholeWithDeltaRi)

# --------------------------------------------------------------------------------------------------
pestWithDeltaRi_TPR = []
pestWithDeltaRi_FNR = []
pestWithDeltaRi_FDR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
for temp in Array(inputDB_PestWithDeltaRi[:, "p(1)"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR, FDR = get1rate(inputDB_PestWithDeltaRi, prob)
        push!(pestWithDeltaRi_TPR, TPR)
        push!(pestWithDeltaRi_FNR, FNR)
        push!(pestWithDeltaRi_FDR, FDR)
    else
        push!(pestWithDeltaRi_TPR, TPR)
        push!(pestWithDeltaRi_FNR, FNR)
        push!(pestWithDeltaRi_FDR, FDR)
    end
end

inputDB_PestWithDeltaRi[!, "TPR"] = pestWithDeltaRi_TPR
inputDB_PestWithDeltaRi[!, "FNR"] = pestWithDeltaRi_FNR
inputDB_PestWithDeltaRi[!, "FDR"] = pestWithDeltaRi_FDR

# save, ouputing df 4103848 x 23+3 df 
#savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_Pest0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_Pest0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)

# ==================================================================================================
# plot P(1)threshold-to-TPR & P(1)threshold-to-TNR
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()

describe(inputDB_WholeWithDeltaRi)[end-4:end, :]

outplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)

plot!(inputDB_WholeWithDeltaRi[:, end-4], [inputDB_WholeWithDeltaRi[:, end-2] inputDB_WholeWithDeltaRi[:, end-1]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) threshold", 
        label = ["True positive rate" "False negative rate"], 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = :left, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)

plot!(inputDB_WholeWithDeltaRi[:, end], inputDB_WholeWithDeltaRi[:, end-2], 
        subplot = 2, framestyle = :box, 
        xlabel = "False discovery rate", 
        xguidefontsize=12, 
        ylabel = "True positive rate", 
        yguidefontsize=12, 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = false, 
        size = (1200,600), 
        dpi = 300)

#savefig(outplotP1toRate, "F:\\UvA\\TPTNPrediction_P1threshold2TPRFNRFDR_Whole0d5FinalScoreRatio.png")
savefig(outplotP1toRate, "F:\\UvA\\TPTNPrediction_P1threshold2TPRFNRFDR_Whole0d5FinalScoreRatioDE.png")

# --------------------------------------------------------------------------------------------------
# plot P(1)threshold-to-TPR & P(1)threshold-to-TNR
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()

describe(inputDB_PestWithDeltaRi)[end-4:end, :]

outplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)

plot!(inputDB_PestWithDeltaRi[:, end-4], [inputDB_PestWithDeltaRi[:, end-2] inputDB_PestWithDeltaRi[:, end-1]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) threshold", 
        label = ["True positive rate" "False negative rate"], 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = :left, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)

plot!(inputDB_PestWithDeltaRi[:, end], inputDB_PestWithDeltaRi[:, end-2], 
        subplot = 2, framestyle = :box, 
        xlabel = "False discovery rate", 
        xguidefontsize=12, 
        ylabel = "True positive rate", 
        yguidefontsize=12, 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = false, 
        size = (1200,600), 
        dpi = 300)

#savefig(outplotP1toRate, "F:\\UvA\\TPTNPrediction_P1threshold2TPRFNRFDR_Pest0d5FinalScoreRatio.png")
savefig(outplotP1toRate, "F:\\UvA\\TPTNPrediction_P1threshold2TPRFNRFDR_Pest0d5FinalScoreRatioDE.png")
