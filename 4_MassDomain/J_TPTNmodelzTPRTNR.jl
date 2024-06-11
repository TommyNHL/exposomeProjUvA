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
# inputing 1636788 x 24 df; 409002 x 24 df; 61988 x 21 df
inputDB_TrainWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv", DataFrame)
inputDB_TrainWithoutDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv", DataFrame)
inputDB_TestWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv", DataFrame)
inputDB_TestWithoutDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv", DataFrame)
inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv", DataFrame)
inputDB_PestWithoutDeltaRi = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv", DataFrame)

# ==================================================================================================
# prepare plotting confusion matrix
# --------------------------------------------------------------------------------------------------
inputDB_TrainWithDeltaRi[!, "CM"] .= String("")
inputDB_TrainWithDeltaRi_TP = 0
inputDB_TrainWithDeltaRi_FP = 0
inputDB_TrainWithDeltaRi_TN = 0
inputDB_TrainWithDeltaRi_FN = 0
for i in 1:size(inputDB_TrainWithDeltaRi , 1)
    if (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_TrainWithDeltaRi[i, "CM"] = "TP"
        inputDB_TrainWithDeltaRi_TP += 1
    elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_TrainWithDeltaRi[i, "CM"] = "FP"
        inputDB_TrainWithDeltaRi_FP += 1
    elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
        inputDB_TrainWithDeltaRi[i, "CM"] = "TN"
        inputDB_TrainWithDeltaRi_TN += 1
    elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
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

# save, ouputing df 1636788 x 24+1 df 
savePath = "F:\\UvA\\dataframePostPredict_TrainWithDeltaRi_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_TrainWithDeltaRi)

# --------------------------------------------------------------------------------------------------
inputDB_TrainWithoutDeltaRi[!, "CM"] .= String("")
inputDB_TrainWithoutDeltaRi_TP = 0
inputDB_TrainWithoutDeltaRi_FP = 0
inputDB_TrainWithoutDeltaRi_TN = 0
inputDB_TrainWithoutDeltaRi_FN = 0
for i in 1:size(inputDB_TrainWithoutDeltaRi , 1)
    if (inputDB_TrainWithoutDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
        inputDB_TrainWithoutDeltaRi[i, "CM"] = "TP"
        inputDB_TrainWithoutDeltaRi_TP += 1
    elseif (inputDB_TrainWithoutDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
        inputDB_TrainWithoutDeltaRi[i, "CM"] = "FP"
        inputDB_TrainWithoutDeltaRi_FP += 1
    elseif (inputDB_TrainWithoutDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
        inputDB_TrainWithoutDeltaRi[i, "CM"] = "TN"
        inputDB_TrainWithoutDeltaRi_TN += 1
    elseif (inputDB_TrainWithoutDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
        inputDB_TrainWithoutDeltaRi[i, "CM"] = "FN"
        inputDB_TrainWithoutDeltaRi_FN += 1
    end
end
describe(inputDB_TrainWithoutDeltaRi)[end-5:end, :]

CM_TrainWithout = zeros(2, 2)
CM_TrainWithout[2, 1] = inputDB_TrainWithoutDeltaRi_TP
CM_TrainWithout[2, 2] = inputDB_TrainWithoutDeltaRi_FP
CM_TrainWithout[1, 2] = inputDB_TrainWithoutDeltaRi_TN
CM_TrainWithout[1, 1] = inputDB_TrainWithoutDeltaRi_FN

# save, ouputing df 1636788 x 24+1 df 
savePath = "F:\\UvA\\dataframePostPredict_TrainWithoutDeltaRi_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_TrainWithoutDeltaRi)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
inputDB_TestWithDeltaRi[!, "CM"] .= String("")
inputDB_TestWithDeltaRi_TP = 0
inputDB_TestWithDeltaRi_FP = 0
inputDB_TestWithDeltaRi_TN = 0
inputDB_TestWithDeltaRi_FN = 0
for i in 1:size(inputDB_TestWithDeltaRi , 1)
    if (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_TestWithDeltaRi[i, "CM"] = "TP"
        inputDB_TestWithDeltaRi_TP += 1
    elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
        inputDB_TestWithDeltaRi[i, "CM"] = "FP"
        inputDB_TestWithDeltaRi_FP += 1
    elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
        inputDB_TestWithDeltaRi[i, "CM"] = "TN"
        inputDB_TestWithDeltaRi_TN += 1
    elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
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

# save, ouputing df 409002 x 24+1 df 
savePath = "F:\\UvA\\dataframePostPredict_TestWithDeltaRi_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_TestWithDeltaRi)

# --------------------------------------------------------------------------------------------------
inputDB_TestWithoutDeltaRi[!, "CM"] .= String("")
inputDB_TestWithoutDeltaRi_TP = 0
inputDB_TestWithoutDeltaRi_FP = 0
inputDB_TestWithoutDeltaRi_TN = 0
inputDB_TestWithoutDeltaRi_FN = 0
for i in 1:size(inputDB_TestWithoutDeltaRi , 1)
    if (inputDB_TestWithoutDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
        inputDB_TestWithoutDeltaRi[i, "CM"] = "TP"
        inputDB_TestWithoutDeltaRi_TP += 1
    elseif (inputDB_TestWithoutDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
        inputDB_TestWithoutDeltaRi[i, "CM"] = "FP"
        inputDB_TestWithoutDeltaRi_FP += 1
    elseif (inputDB_TestWithoutDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
        inputDB_TestWithoutDeltaRi[i, "CM"] = "TN"
        inputDB_TestWithoutDeltaRi_TN += 1
    elseif (inputDB_TestWithoutDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithoutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
        inputDB_TestWithoutDeltaRi[i, "CM"] = "FN"
        inputDB_TestWithoutDeltaRi_FN += 1
    end
end
describe(inputDB_TestWithoutDeltaRi)[end-5:end, :]

CM_TestWithout = zeros(2, 2)
CM_TestWithout[2, 1] = inputDB_TestWithoutDeltaRi_TP
CM_TestWithout[2, 2] = inputDB_TestWithoutDeltaRi_FP
CM_TestWithout[1, 2] = inputDB_TestWithoutDeltaRi_TN
CM_TestWithout[1, 1] = inputDB_TestWithoutDeltaRi_FN

# save, ouputing df 409002 x 24+1 df 
savePath = "F:\\UvA\\dataframePostPredict_TestWithoutDeltaRi_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_TestWithoutDeltaRi)

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

# save, ouputing df 61988 x 21+1 df 
savePath = "F:\\UvA\\dataframePostPredict_PestWithDeltaRi_new0d5FinalScoreRatio.csv"
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

# save, ouputing df 61988 x 21+1 df 
savePath = "F:\\UvA\\dataframePostPredict_PestWithoutDeltaRi_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_PestWithoutDeltaRi)

# ==================================================================================================
# ==================================================================================================
# plot confusion matrix
layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
default(grid = false, legend = false)
gr()
TrainOutplotCM = plot(layout = layout, link = :both, 
        size = (1400, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_TrainWith, cmap = :viridis, cbar = :true, 
        clims = (5000, 200000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n144,300"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n105,825"], subplot = 2)
        annotate!(["1"], ["0"], ["FN\n49"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n1,386,614"], subplot = 2)
heatmap!(["1", "0"], ["0", "1"], CM_TrainWithout, cmap = :viridis, cbar = :none, 
        clims = (5000, 200000), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n144,260"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n204,310"], subplot = 1)
        annotate!(["1"], ["0"], ["FN\n89"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n1,288,129"], subplot = 1)
savefig(TrainOutplotCM, "F:\\UvA\\TPTNPrediction_RFtrainCM_new0d5FinalScoreRatio.png")

# --------------------------------------------------------------------------------------------------
layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
default(grid = false, legend = false)
gr()
TestOutplotCM = plot(layout = layout, link = :both, 
        size = (1400, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_TestWith, cmap = :viridis, cbar = :true, 
        clims = (1200, 50000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n33,420"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n34,733"], subplot = 2)
        annotate!(["1"], ["0"], ["FN\n2,686"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n338,163"], subplot = 2)
heatmap!(["1", "0"], ["0", "1"], CM_TestWithout, cmap = :viridis, cbar = :none, 
        clims = (1200, 50000), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n33,743"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n57,424"], subplot = 1)
        annotate!(["1"], ["0"], ["FN\n2,363"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n315,472"], subplot = 1)
savefig(TestOutplotCM, "F:\\UvA\\TPTNPrediction_RFtestCM_new0d5FinalScoreRatio.png")

# --------------------------------------------------------------------------------------------------
layout = @layout [a{0.45w,1.0h} b{0.55w,1.0h}]
default(grid = false, legend = false)
gr()
PestOutplotCM = plot(layout = layout, link = :both, 
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
        annotate!(["1"], ["1"], ["TP\n6,882"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n2,005"], subplot = 2, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n1,964"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n51,137"], subplot = 2)
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
        annotate!(["1"], ["1"], ["TP\n7,730"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n2,813"], subplot = 1, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n1,116"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n50,329"], subplot = 1)
savefig(PestOutplotCM, "F:\\UvA\\TPTNPrediction_RFpestCM_new0d5FinalScoreRatio.png")

# ==================================================================================================

# prepare plotting P(TP)threshold-to-TPR curve
# 1636788 x 24+1 df
inputDB_TrainWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_TrainWithDeltaRi_new0d5FinalScoreRatio.csv", DataFrame)
sort!(inputDB_TrainWithDeltaRi, [:"p(1)"], rev = true)
for i in 1:size(inputDB_TrainWithDeltaRi, 1)
    inputDB_TrainWithDeltaRi[i, "p(1)"] = round(float(inputDB_TrainWithDeltaRi[i, "p(1)"]), digits = 2)
end

# 409002 x 24+1 df
inputDB_TestWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_TestWithDeltaRi_new0d5FinalScoreRatio.csv", DataFrame)
sort!(inputDB_TestWithDeltaRi, [:"p(1)"], rev = true)
for i in 1:size(inputDB_TestWithDeltaRi, 1)
    inputDB_TestWithDeltaRi[i, "p(1)"] = round(float(inputDB_TestWithDeltaRi[i, "p(1)"]), digits = 2)
end

# 61988 x 21+1 df
inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\dataframePostPredict_PestWithDeltaRi_new0d5FinalScoreRatio.csv", DataFrame)
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
TrainWithDeltaRi_TPR = []
TrainWithDeltaRi_FNR = []
TrainWithDeltaRi_FDR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
for temp in Array(inputDB_TrainWithDeltaRi[:, "p(1)"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR, FDR = get1rate(inputDB_TrainWithDeltaRi, prob)
        push!(TrainWithDeltaRi_TPR, TPR)
        push!(TrainWithDeltaRi_FNR, FNR)
        push!(TrainWithDeltaRi_FDR, FDR)
    else
        push!(TrainWithDeltaRi_TPR, TPR)
        push!(TrainWithDeltaRi_FNR, FNR)
        push!(TrainWithDeltaRi_FDR, FDR)
    end
end

inputDB_TrainWithDeltaRi[!, "TPR"] = TrainWithDeltaRi_TPR
inputDB_TrainWithDeltaRi[!, "FNR"] = TrainWithDeltaRi_FNR
inputDB_TrainWithDeltaRi[!, "FDR"] = TrainWithDeltaRi_FDR

# save, ouputing df 1636788 x 24+1+3 df 
savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_newTrain0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_TrainWithDeltaRi)

# --------------------------------------------------------------------------------------------------
TestWithDeltaRi_TPR = []
TestWithDeltaRi_FNR = []
TestWithDeltaRi_FDR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
for temp in Array(inputDB_TestWithDeltaRi[:, "p(1)"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR, FDR = get1rate(inputDB_TestWithDeltaRi, prob)
        push!(TestWithDeltaRi_TPR, TPR)
        push!(TestWithDeltaRi_FNR, FNR)
        push!(TestWithDeltaRi_FDR, FDR)
    else
        push!(TestWithDeltaRi_TPR, TPR)
        push!(TestWithDeltaRi_FNR, FNR)
        push!(TestWithDeltaRi_FDR, FDR)
    end
end

inputDB_TestWithDeltaRi[!, "TPR"] = TestWithDeltaRi_TPR
inputDB_TestWithDeltaRi[!, "FNR"] = TestWithDeltaRi_FNR
inputDB_TestWithDeltaRi[!, "FDR"] = TestWithDeltaRi_FDR

# save, ouputing df 409002 x 24+1+3 df 
savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_newTest0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_TestWithDeltaRi)

# --------------------------------------------------------------------------------------------------
PestWithDeltaRi_TPR = []
PestWithDeltaRi_FNR = []
PestWithDeltaRi_FDR = []
prob = -1
TPR = 0
FNR = 0
FDR = 0
for temp in Array(inputDB_PestWithDeltaRi[:, "p(1)"])
    if (temp != prob)
        println(temp)
        prob = temp
        TPR, FNR, FDR = get1rate(inputDB_PestWithDeltaRi, prob)
        push!(PestWithDeltaRi_TPR, TPR)
        push!(PestWithDeltaRi_FNR, FNR)
        push!(PestWithDeltaRi_FDR, FDR)
    else
        push!(PestWithDeltaRi_TPR, TPR)
        push!(PestWithDeltaRi_FNR, FNR)
        push!(PestWithDeltaRi_FDR, FDR)
    end
end

inputDB_PestWithDeltaRi[!, "TPR"] = PestWithDeltaRi_TPR
inputDB_PestWithDeltaRi[!, "FNR"] = PestWithDeltaRi_FNR
inputDB_PestWithDeltaRi[!, "FDR"] = PestWithDeltaRi_FDR

# save, ouputing df 61988 x 21+1+3 df 
savePath = "F:\\UvA\\dataframePostPredict_TPRFNRFDR_newPest0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)

# ==================================================================================================
# plot P(1)threshold-to-TPR & P(1)threshold-to-TNR
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()

describe(inputDB_TrainWithDeltaRi)[end-4:end, :]

TrainOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)

plot!(inputDB_TrainWithDeltaRi[:, end-4], [inputDB_TrainWithDeltaRi[:, end-2] inputDB_TrainWithDeltaRi[:, end-1]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) threshold", 
        label = ["True positive rate" "False negative rate"], 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = :left, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)
        new_xticks = ([0.865], ["\$\\bar"])
        vline!(new_xticks[1], label = "5% FDR cutoff at P(1) = 0.865", legendfont = font(10), lc = "red", subplot = 1)

plot!(inputDB_TrainWithDeltaRi[:, end], inputDB_TrainWithDeltaRi[:, end-2], 
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
        new_xticks = ([0.05], ["\$\\bar"])
        vline!(new_xticks[1], label = "5% FDR", lc = "red", subplot = 2)
savefig(TrainOutplotP1toRate, "F:\\UvA\\TPTNPrediction_P1threshold2TPRFNRFDR_newTrain0d5FinalScoreRatio.png")

# --------------------------------------------------------------------------------------------------
# plot P(1)threshold-to-TPR & P(1)threshold-to-TNR
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()

describe(inputDB_TestWithDeltaRi)[end-4:end, :]

TestOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)

plot!(inputDB_TestWithDeltaRi[:, end-4], [inputDB_TestWithDeltaRi[:, end-2] inputDB_TestWithDeltaRi[:, end-1]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) threshold", 
        label = ["True positive rate" "False negative rate"], 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = :left, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)
        new_xticks = ([0.965], ["\$\\bar"])
        vline!(new_xticks[1], label = "5% FDR cutoff at P(1) = 0.965", legendfont = font(10), lc = "red", subplot = 1)

plot!(inputDB_TestWithDeltaRi[:, end], inputDB_TestWithDeltaRi[:, end-2], 
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
        new_xticks = ([0.05], ["\$\\bar"])
        vline!(new_xticks[1], label = "5% FDR", lc = "red", subplot = 2)
savefig(TestOutplotP1toRate, "F:\\UvA\\TPTNPrediction_P1threshold2TPRFNRFDR_newTest0d5FinalScoreRatio.png")

# --------------------------------------------------------------------------------------------------
# plot P(1)threshold-to-TPR & P(1)threshold-to-TNR
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()

describe(inputDB_PestWithDeltaRi)[end-4:end, :]

PestOutplotP1toRate = plot(layout = layout, link = :both, 
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
        new_xticks = ([0.935], ["\$\\bar"])
        vline!(new_xticks[1], label = "5% FDR cutoff at P(1) = 0.935", legendfont = font(10), lc = "red", subplot = 1)

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
        new_xticks = ([0.05], ["\$\\bar"])
        vline!(new_xticks[1], label = "5% FDR", lc = "red", subplot = 2)
savefig(PestOutplotP1toRate, "F:\\UvA\\TPTNPrediction_P1threshold2TPRFNRFDR_newPest0d5FinalScoreRatio.png")
