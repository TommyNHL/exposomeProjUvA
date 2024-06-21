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
# inputing 485631 x 25 df; 121946 x 25 df; 4757 x 22 df
inputDB_TrainWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDFall_withDeltaRIandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)

inputDB_TestWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDFall_withDeltaRIandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)

inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDFreal_withDeltaRIandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)


# ==================================================================================================
# prepare plotting confusion matrix
# --------------------------------------------------------------------------------------------------
inputDB_TrainWithDeltaRi[!, "CM"] .= String("")
inputDB_TrainWithDeltaRi_TP = 0
inputDB_TrainWithDeltaRi_FP = 0
inputDB_TrainWithDeltaRi_TN = 0
inputDB_TrainWithDeltaRi_FN = 0
for i in 1:size(inputDB_TrainWithDeltaRi , 1)
    if (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 1)
        inputDB_TrainWithDeltaRi[i, "CM"] = "TP"
        inputDB_TrainWithDeltaRi_TP += 1
    elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 1)
        inputDB_TrainWithDeltaRi[i, "CM"] = "FP"
        inputDB_TrainWithDeltaRi_FP += 1
    elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 0)
        inputDB_TrainWithDeltaRi[i, "CM"] = "TN"
        inputDB_TrainWithDeltaRi_TN += 1
    elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 0)
        inputDB_TrainWithDeltaRi[i, "CM"] = "FN"
        inputDB_TrainWithDeltaRi_FN += 1
    end
end
describe(inputDB_TrainWithDeltaRi)[end-5:end, :]

CM_TrainWith = zeros(2, 2)
CM_TrainWith[2, 1] = inputDB_TrainWithDeltaRi_TP
CM_TrainWith[2, 2] = inputDB_TrainWithDeltaRi_FP
CM_TrainWith[1, 2] = inputDB_TrainWithDeltaRi_TN
CM_TrainWith[1, 1] = inputDB_TrainWithDeltaRi_FN

# save, ouputing df 485631 x 25+1 df 
savePath = "F:\\UvA\\dataframePostPredict_TrainALLWithDeltaRI_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB_TrainWithDeltaRi)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
inputDB_TestWithDeltaRi[!, "CM"] .= String("")
inputDB_TestWithDeltaRi_TP = 0
inputDB_TestWithDeltaRi_FP = 0
inputDB_TestWithDeltaRi_TN = 0
inputDB_TestWithDeltaRi_FN = 0
for i in 1:size(inputDB_TestWithDeltaRi , 1)
    if (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 1)
        inputDB_TestWithDeltaRi[i, "CM"] = "TP"
        inputDB_TestWithDeltaRi_TP += 1
    elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 1)
        inputDB_TestWithDeltaRi[i, "CM"] = "FP"
        inputDB_TestWithDeltaRi_FP += 1
    elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 0)
        inputDB_TestWithDeltaRi[i, "CM"] = "TN"
        inputDB_TestWithDeltaRi_TN += 1
    elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 0)
        inputDB_TestWithDeltaRi[i, "CM"] = "FN"
        inputDB_TestWithDeltaRi_FN += 1
    end
end
describe(inputDB_TestWithDeltaRi)[end-5:end, :]

CM_TestWith = zeros(2, 2)
CM_TestWith[2, 1] = inputDB_TestWithDeltaRi_TP
CM_TestWith[2, 2] = inputDB_TestWithDeltaRi_FP
CM_TestWith[1, 2] = inputDB_TestWithDeltaRi_TN
CM_TestWith[1, 1] = inputDB_TestWithDeltaRi_FN

# save, ouputing df 121946 x 25+1 df 
savePath = "F:\\UvA\\dataframePostPredict_TestALLWithDeltaRI_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB_TestWithDeltaRi)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
inputDB_PestWithDeltaRi[!, "CM"] .= String("")
inputDB_PestWithDeltaRi_TP = 0
inputDB_PestWithDeltaRi_FP = 0
inputDB_PestWithDeltaRi_TN = 0
inputDB_PestWithDeltaRi_FN = 0
for i in 1:size(inputDB_PestWithDeltaRi , 1)
    if (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 1)
        inputDB_PestWithDeltaRi[i, "CM"] = "TP"
        inputDB_PestWithDeltaRi_TP += 1
    elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 1)
        inputDB_PestWithDeltaRi[i, "CM"] = "FP"
        inputDB_PestWithDeltaRi_FP += 1
    elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 0)
        inputDB_PestWithDeltaRi[i, "CM"] = "TN"
        inputDB_PestWithDeltaRi_TN += 1
    elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "withDeltaRIpredictTPTN"] == 0)
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

# save, ouputing df 4757 x 22+1 df 
savePath = "F:\\UvA\\dataframePostPredict_PestREALWithDeltaRI_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)


# ==================================================================================================
# ==================================================================================================
# plot confusion matrix
layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
default(grid = false, legend = false)
gr()
TrainOutplotCM = plot(layout = layout, link = :both, 
        size = (1400, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_TrainWith, cmap = :viridis, cbar = :none, 
        clims = (0, 121408), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Training Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n105,819"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n105,403"], subplot = 1)
        annotate!(["1"], ["0"], ["FN\n45,491"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n228,918"], subplot = 1)
heatmap!(["1", "0"], ["0", "1"], CM_TestWith, cmap = :viridis, cbar = :true, 
        clims = (0, 30487), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Testing Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n25,576"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n27,477"], subplot = 2)
        annotate!(["1"], ["0"], ["FN\n12,389"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n56,504"], subplot = 2)
savefig(TrainOutplotCM, "F:\\UvA\\TPTNPrediction_DTtrainTestCM_withhl0d5FinalScoreRatio2DE2Filter.png")

# --------------------------------------------------------------------------------------------------

TestOutplotCM = heatmap(["1", "0"], ["0", "1"], CM_PestWith, cmap = :viridis, cbar = :true, margin = (10, :mm),
        clims = (0, 1189), 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Real Sample Dataset", 
        titlefont = font(16), 
        size = (700,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n717"])
        annotate!(["0"], ["1"], ["FP\n1,437"])
        annotate!(["1"], ["0"], ["FN\n617"])
        annotate!(["0"], ["0"], ["TN\n1,986"])
savefig(TestOutplotCM, "F:\\UvA\\TPTNPrediction_DTpestCM_withhl0d5FinalScoreRatio2DE2Filter.png")


# ==================================================================================================

# prepare plotting P(TP)threshold-to-TPR curve
# 485631 x 25+1 df
inputDB_TrainWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_TrainALLWithDeltaRI_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)
sort!(inputDB_TrainWithDeltaRi, [:"p(1)"], rev = true)
for i in 1:size(inputDB_TrainWithDeltaRi, 1)
    inputDB_TrainWithDeltaRi[i, "p(1)"] = round(float(inputDB_TrainWithDeltaRi[i, "p(1)"]), digits = 2)
end

# 121946 x 25+1 df
inputDB_TestWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_TestALLWithDeltaRI_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)
sort!(inputDB_TestWithDeltaRi, [:"p(1)"], rev = true)
for i in 1:size(inputDB_TestWithDeltaRi, 1)
    inputDB_TestWithDeltaRi[i, "p(1)"] = round(float(inputDB_TestWithDeltaRi[i, "p(1)"]), digits = 2)
end

# 4757 x 22+1 df
inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_PestREALWithDeltaRI_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)
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
    return (TP / (TP + FN)), (FN / (TP + FN)), (FP / (FP + TP)), (FP / (FP + TN)), (TN / (TN + FP))
end

# --------------------------------------------------------------------------------------------------
TrainWithDeltaRi_TPR = []
TrainWithDeltaRi_FNR = []
TrainWithDeltaRi_FDR = []
TrainWithDeltaRi_FPR = []
TrainWithDeltaRi_TNR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
FPR = 0
TNR = 0
for temp in Array(inputDB_TrainWithDeltaRi[:, "p(1)"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR, FDR, FPR, TNR = get1rate(inputDB_TrainWithDeltaRi, prob)
        push!(TrainWithDeltaRi_TPR, TPR)
        push!(TrainWithDeltaRi_FNR, FNR)
        push!(TrainWithDeltaRi_FDR, FDR)
        push!(TrainWithDeltaRi_FPR, FPR)
        push!(TrainWithDeltaRi_TNR, TNR)
    else
        push!(TrainWithDeltaRi_TPR, TPR)
        push!(TrainWithDeltaRi_FNR, FNR)
        push!(TrainWithDeltaRi_FDR, FDR)
        push!(TrainWithDeltaRi_FPR, FPR)
        push!(TrainWithDeltaRi_TNR, TNR)
    end
end

inputDB_TrainWithDeltaRi[!, "TPR"] = TrainWithDeltaRi_TPR
inputDB_TrainWithDeltaRi[!, "FNR"] = TrainWithDeltaRi_FNR
inputDB_TrainWithDeltaRi[!, "FDR"] = TrainWithDeltaRi_FDR
inputDB_TrainWithDeltaRi[!, "FPR"] = TrainWithDeltaRi_FPR
inputDB_TrainWithDeltaRi[!, "TNR"] = TrainWithDeltaRi_TNR

# save, ouputing df 485631 x 25+1+4 df 
savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_newTrainALL_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB_TrainWithDeltaRi)

# --------------------------------------------------------------------------------------------------
TestWithDeltaRi_TPR = []
TestWithDeltaRi_FNR = []
TestWithDeltaRi_FDR = []
TestWithDeltaRi_FPR = []
TestWithDeltaRi_TNR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
FPR = 0
TNR = 0
for temp in Array(inputDB_TestWithDeltaRi[:, "p(1)"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR, FDR, FPR, TNR = get1rate(inputDB_TestWithDeltaRi, prob)
        push!(TestWithDeltaRi_TPR, TPR)
        push!(TestWithDeltaRi_FNR, FNR)
        push!(TestWithDeltaRi_FDR, FDR)
        push!(TestWithDeltaRi_FPR, FPR)
        push!(TestWithDeltaRi_TNR, TNR)
    else
        push!(TestWithDeltaRi_TPR, TPR)
        push!(TestWithDeltaRi_FNR, FNR)
        push!(TestWithDeltaRi_FDR, FDR)
        push!(TestWithDeltaRi_FPR, FPR)
        push!(TestWithDeltaRi_TNR, TNR)
    end
end

inputDB_TestWithDeltaRi[!, "TPR"] = TestWithDeltaRi_TPR
inputDB_TestWithDeltaRi[!, "FNR"] = TestWithDeltaRi_FNR
inputDB_TestWithDeltaRi[!, "FDR"] = TestWithDeltaRi_FDR
inputDB_TestWithDeltaRi[!, "FPR"] = TestWithDeltaRi_FPR
inputDB_TestWithDeltaRi[!, "TNR"] = TestWithDeltaRi_TNR

# save, ouputing df 121946 x 25+1+3 df 
savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_newTestALL_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB_TestWithDeltaRi)

# --------------------------------------------------------------------------------------------------
PestWithDeltaRi_TPR = []
PestWithDeltaRi_FNR = []
PestWithDeltaRi_FDR = []
PestWithDeltaRi_FPR = []
PestWithDeltaRi_TNR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
FPR = 0
TNR = 0
for temp in Array(inputDB_PestWithDeltaRi[:, "p(1)"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR, FDR, FPR, TNR = get1rate(inputDB_PestWithDeltaRi, prob)
        push!(PestWithDeltaRi_TPR, TPR)
        push!(PestWithDeltaRi_FNR, FNR)
        push!(PestWithDeltaRi_FDR, FDR)
        push!(PestWithDeltaRi_FPR, FPR)
        push!(PestWithDeltaRi_TNR, TNR)
    else
        push!(PestWithDeltaRi_TPR, TPR)
        push!(PestWithDeltaRi_FNR, FNR)
        push!(PestWithDeltaRi_FDR, FDR)
        push!(PestWithDeltaRi_FPR, FPR)
        push!(PestWithDeltaRi_TNR, TNR)
    end
end

inputDB_PestWithDeltaRi[!, "TPR"] = PestWithDeltaRi_TPR
inputDB_PestWithDeltaRi[!, "FNR"] = PestWithDeltaRi_FNR
inputDB_PestWithDeltaRi[!, "FDR"] = PestWithDeltaRi_FDR
inputDB_PestWithDeltaRi[!, "FPR"] = PestWithDeltaRi_FPR
inputDB_PestWithDeltaRi[!, "TNR"] = PestWithDeltaRi_TNR

# save, ouputing df 4757 x 22+1+3 df 
savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_newPestREAL_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)

# ==================================================================================================
# plot P(1)threshold-to-TPR & P(1)threshold-to-TNR
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()

describe(inputDB_TrainWithDeltaRi)[end-6:end, :]

TrainOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)

plot!(inputDB_TrainWithDeltaRi[:, end-6], [inputDB_TrainWithDeltaRi[:, end-4] inputDB_TrainWithDeltaRi[:, end-3]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        label = ["True Positive Rate" "False Negative Rate"], 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = :left, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)

plot!(inputDB_TrainWithDeltaRi[:, end-6], inputDB_TrainWithDeltaRi[:, end-2], 
        subplot = 2, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=12, 
        label = "False Discovery Rate", 
        yguidefontsize=12, 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = :best, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)
        new_yticks = ([0.10], ["\$\\bar"])
        hline!(new_yticks[1], label = "10% FDR Cutoff at P(1) = 0.92", legendfont = font(10), lc = "red", subplot = 2)
savefig(TrainOutplotP1toRate, "F:\\UvA\\TPTNPrediction_P1threshold2TPRFNRFDR_newTrainALL_withhl0d5FinalScoreRatio2DE2Filter_DT.png")

# --------------------------------------------------------------------------------------------------
inputDB_TrainWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDFall_withDeltaRIandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)

inputDB_TestWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDFall_withDeltaRIandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)

inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDFreal_withDeltaRIandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)
# ==================================================================================================
# prepare plotting confusion matrix again at 0.96
# --------------------------------------------------------------------------------------------------
inputDB_TrainWithDeltaRi[!, "CM"] .= String("")
inputDB_TrainWithDeltaRi_TP = 0
inputDB_TrainWithDeltaRi_FP = 0
inputDB_TrainWithDeltaRi_TN = 0
inputDB_TrainWithDeltaRi_FN = 0
for i in 1:size(inputDB_TrainWithDeltaRi , 1)
    if (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "p(1)"] >= 0.92)
        inputDB_TrainWithDeltaRi[i, "CM"] = "TP"
        inputDB_TrainWithDeltaRi_TP += 1
    elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "p(1)"] >= 0.92)
        inputDB_TrainWithDeltaRi[i, "CM"] = "FP"
        inputDB_TrainWithDeltaRi_FP += 1
    elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "p(1)"] < 0.92)
        inputDB_TrainWithDeltaRi[i, "CM"] = "TN"
        inputDB_TrainWithDeltaRi_TN += 1
    elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "p(1)"] < 0.92)
        inputDB_TrainWithDeltaRi[i, "CM"] = "FN"
        inputDB_TrainWithDeltaRi_FN += 1
    end
end
describe(inputDB_TrainWithDeltaRi)[end-5:end, :]

CM_TrainWith = zeros(2, 2)
CM_TrainWith[2, 1] = inputDB_TrainWithDeltaRi_TP
CM_TrainWith[2, 2] = inputDB_TrainWithDeltaRi_FP
CM_TrainWith[1, 2] = inputDB_TrainWithDeltaRi_TN
CM_TrainWith[1, 1] = inputDB_TrainWithDeltaRi_FN

# save, ouputing df 485631 x 25+1 df 
savePath = "F:\\UvA\\dataframePostPredict_TrainALLWithDeltaRI_withhl0d5FinalScoreRatio2DE2Filter_DTcutoff.csv"
CSV.write(savePath, inputDB_TrainWithDeltaRi)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
inputDB_TestWithDeltaRi[!, "CM"] .= String("")
inputDB_TestWithDeltaRi_TP = 0
inputDB_TestWithDeltaRi_FP = 0
inputDB_TestWithDeltaRi_TN = 0
inputDB_TestWithDeltaRi_FN = 0
for i in 1:size(inputDB_TestWithDeltaRi , 1)
    if (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "p(1)"] >= 0.92)
        inputDB_TestWithDeltaRi[i, "CM"] = "TP"
        inputDB_TestWithDeltaRi_TP += 1
    elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "p(1)"] >= 0.92)
        inputDB_TestWithDeltaRi[i, "CM"] = "FP"
        inputDB_TestWithDeltaRi_FP += 1
    elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "p(1)"] < 0.92)
        inputDB_TestWithDeltaRi[i, "CM"] = "TN"
        inputDB_TestWithDeltaRi_TN += 1
    elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "p(1)"] < 0.92)
        inputDB_TestWithDeltaRi[i, "CM"] = "FN"
        inputDB_TestWithDeltaRi_FN += 1
    end
end
describe(inputDB_TestWithDeltaRi)[end-5:end, :]

CM_TestWith = zeros(2, 2)
CM_TestWith[2, 1] = inputDB_TestWithDeltaRi_TP
CM_TestWith[2, 2] = inputDB_TestWithDeltaRi_FP
CM_TestWith[1, 2] = inputDB_TestWithDeltaRi_TN
CM_TestWith[1, 1] = inputDB_TestWithDeltaRi_FN

# save, ouputing df 121946 x 25+1 df 
savePath = "F:\\UvA\\dataframePostPredict_TestALLWithDeltaRI_withhl0d5FinalScoreRatio2DE2Filter_DTcutoff.csv"
CSV.write(savePath, inputDB_TestWithDeltaRi)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
inputDB_PestWithDeltaRi[!, "CM"] .= String("")
inputDB_PestWithDeltaRi_TP = 0
inputDB_PestWithDeltaRi_FP = 0
inputDB_PestWithDeltaRi_TN = 0
inputDB_PestWithDeltaRi_FN = 0
for i in 1:size(inputDB_PestWithDeltaRi , 1)
    if (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "p(1)"] >= 0.92)
        inputDB_PestWithDeltaRi[i, "CM"] = "TP"
        inputDB_PestWithDeltaRi_TP += 1
    elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "p(1)"] >= 0.92)
        inputDB_PestWithDeltaRi[i, "CM"] = "FP"
        inputDB_PestWithDeltaRi_FP += 1
    elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "p(1)"] < 0.92)
        inputDB_PestWithDeltaRi[i, "CM"] = "TN"
        inputDB_PestWithDeltaRi_TN += 1
    elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "p(1)"] < 0.92)
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

# save, ouputing df 4757 x 22+1 df 
savePath = "F:\\UvA\\dataframePostPredict_PestREALWithDeltaRI_withhl0d5FinalScoreRatio2DE2Filter_DTcutoff.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)


# ==================================================================================================
# ==================================================================================================
# plot confusion matrix
layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
default(grid = false, legend = false)
gr()
TrainOutplotCM = plot(layout = layout, link = :both, 
        size = (1400, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_TrainWith, cmap = :viridis, cbar = :none, 
        clims = (0, 121408), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Training Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["10% FDR\nControlled TP\n7,830"], subplot = 1, font(color="white"))
        annotate!(["0"], ["1"], ["10% FDR\nControlled FP\n815"], subplot = 1, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n143,480"], subplot = 1)
        annotate!(["0"], ["0"], ["TN\n333,506"], subplot = 1)
heatmap!(["1", "0"], ["0", "1"], CM_TestWith, cmap = :viridis, cbar = :true, 
        clims = (0, 30487), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Testing Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["10% FDR\nControlled TP\n1,851"], subplot = 2, font(color="white"))
        annotate!(["0"], ["1"], ["10% FDR\nControlled FP\n260"], subplot = 2, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n36,114"], subplot = 2)
        annotate!(["0"], ["0"], ["TN\n83,721"], subplot = 2)
savefig(TrainOutplotCM, "F:\\UvA\\TPTNPrediction_DTtrainTestCM_withhl0d5FinalScoreRatio2DE2Filter_Cutoff.png")

# --------------------------------------------------------------------------------------------------

TestOutplotCM = heatmap(["1", "0"], ["0", "1"], CM_PestWith, cmap = :viridis, cbar = :true, margin = (10, :mm),
        clims = (0, 1189), 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Real Sample Dataset", 
        titlefont = font(16), 
        size = (700,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["10% FDR\nControlled TP\n8"], font(color="white"))
        annotate!(["0"], ["1"], ["10% FDR\nControlled FP\n18"], font(color="white"))
        annotate!(["1"], ["0"], ["FN\n1,326"])
        annotate!(["0"], ["0"], ["TN\n3,405"])
savefig(TestOutplotCM, "F:\\UvA\\TPTNPrediction_DTpestCM_withhl0d5FinalScoreRatio2DE2Filter_Cutoff.png")
