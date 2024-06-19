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

jl = pyimport("joblib")             # used for loading models
f1_score = pyimport("sklearn.metrics").f1_score
matthews_corrcoef = pyimport("sklearn.metrics").matthews_corrcoef
make_scorer = pyimport("sklearn.metrics").make_scorer
f1 = make_scorer(f1_score, pos_label=1, average="binary")

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: GradientBoostingClassifier
@sk_import linear_model: LogisticRegression
@sk_import ensemble: RandomForestClassifier
@sk_import metrics: recall_score
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

describe((inputDB_test))[1:14, :]
# inputing 820519 x 4+8+1+2+1+1+2+2 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestYesDFwithhl.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
insertcols!(inputDB_test, 10, ("MatchDiff"=>float(0)))
inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
describe(inputDB_test[inputDB_test.LABEL .== 1, :])
inputDB_test = inputDB_test[inputDB_test.MS1Error .>= float(-0.001), :]
inputDB_test = inputDB_test[inputDB_test.MS1Error .<= float(0.001), :]
for i = 1:size(inputDB_test, 1)
    inputDB_test[i, "RefMatchFragRatio"] = log10(inputDB_test[i, "RefMatchFragRatio"])
    inputDB_test[i, "UsrMatchFragRatio"] = log10(inputDB_test[i, "UsrMatchFragRatio"])
    inputDB_test[i, "FinalScoreRatio"] = log10(inputDB_test[i, "FinalScoreRatio"])
    inputDB_test[i, "MatchDiff"] = inputDB_test[i, "DirectMatch"] - inputDB_test[i, "ReversMatch"]
end
# save, ouputing 119054 x 21+1 df, 0:83052; 1:36002 = 0.7167; 1.6534
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDFwithhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB_test)
inputDB_test[inputDB_test.LABEL .== 1, :]

# inputing 3282022 x 4+8+1+2+1+1+2+2 df
inputDB = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainYesDFwithhl.csv", DataFrame)
sort!(inputDB, [:ENTRY])
insertcols!(inputDB, 10, ("MatchDiff"=>float(0)))
inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
describe(inputDB[inputDB.LABEL .== 1, :])
inputDB = inputDB[inputDB.MS1Error .>= float(-0.001), :]
inputDB = inputDB[inputDB.MS1Error .<= float(0.001), :]
for i = 1:size(inputDB, 1)
    inputDB[i, "RefMatchFragRatio"] = log10(inputDB[i, "RefMatchFragRatio"])
    inputDB[i, "UsrMatchFragRatio"] = log10(inputDB[i, "UsrMatchFragRatio"])
    inputDB[i, "FinalScoreRatio"] = log10(inputDB[i, "FinalScoreRatio"])
    inputDB[i, "MatchDiff"] = inputDB[i, "DirectMatch"] - inputDB[i, "ReversMatch"]
end
# save, ouputing 475245 x 21+1 df, 0:330818; 1:144427 = 0.7183; 1.6453
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB)
inputDB[inputDB.LABEL .== 1, :]

# 594299 x 22 df; 
# 119054+475245= 594299, 0:413870; 1:180429 = 0.7180; 1.6469
inputDBInputDB_test = vcat(inputDB, inputDB_test)
sort!(inputDBInputDB_test, [:ENTRY])
inputDBInputDB_test[inputDBInputDB_test.LABEL .== 1, :]

# 136678 x 18 df
inputDB_pest = CSV.read("F:\\UvA\\dataframeTPTNModeling_pest.csv", DataFrame)
sort!(inputDB_pest, [:ENTRY])
insertcols!(inputDB_pest, 10, ("MatchDiff"=>float(0)))
inputDB_pest = inputDB_pest[inputDB_pest.FinalScoreRatio .>= float(0.5), :]
inputDB_pest = inputDB_pest[inputDB_pest.Leverage .<= 0.14604417882015916, :]
describe(inputDB_pest[inputDB_pest.LABEL .== 1, :])
inputDB_pest = inputDB_pest[inputDB_pest.MS1Error .>= float(-0.001), :]
inputDB_pest = inputDB_pest[inputDB_pest.MS1Error .<= float(0.001), :]
for i = 1:size(inputDB_pest, 1)
    inputDB_pest[i, "RefMatchFragRatio"] = log10(inputDB_pest[i, "RefMatchFragRatio"])
    inputDB_pest[i, "UsrMatchFragRatio"] = log10(inputDB_pest[i, "UsrMatchFragRatio"])
    inputDB_pest[i, "FinalScoreRatio"] = log10(inputDB_pest[i, "FinalScoreRatio"])
    inputDB_pest[i, "MatchDiff"] = inputDB_pest[i, "DirectMatch"] - inputDB_pest[i, "ReversMatch"]
end
# save, ouputing 13278 x 18+1 df, 0:4432; 1:8846 = 1.4980; 0.7505
savePath = "F:\\UvA\\dataframeTPTNModeling_pestDFwithhl0d5FinalScore2RatioDE2Filter.csv"
CSV.write(savePath, inputDB_pest)
inputDB_pest[inputDB_pest.LABEL .== 1, :]

# performace
## Maximum absolute error
## mean square error (MSE) calculation
## Root mean square error (RMSE) calculation
function errorDetermination(arrRi, predictedRi)
    sumAE = 0
    maxAE = 0
    for i = 1:size(predictedRi, 1)
        AE = abs(arrRi[i] - predictedRi[i])
        if (AE > maxAE)
            maxAE = AE
        end
        sumAE += (AE ^ 2)
    end
    MSE = sumAE / size(predictedRi, 1)
    RMSE = MSE ^ 0.5
    return maxAE, MSE, RMSE
end

## R-square value
function rSquareDetermination(arrRi, predictedRi)
    sumY = 0
    for i = 1:size(predictedRi, 1)
        sumY += predictedRi[i]
    end
    meanY = sumY / size(predictedRi, 1)
    sumAE = 0
    sumRE = 0
    for i = 1:size(predictedRi, 1)
        AE = abs(arrRi[i] - predictedRi[i])
        RE = abs(arrRi[i] - meanY)
        sumAE += (AE ^ 2)
        sumRE += (RE ^ 2)
    end
    rSquare = 1 - (sumAE / sumRE)
    return rSquare
end

## Average score
function avgScore(arrAcc, cv)
    sumAcc = 0
    for acc in arrAcc
        sumAcc += acc
    end
    return sumAcc / cv
end

# modeling, 5 x 4 x 5 x 9 = 225 times
describe((inputDB))[vcat(5,6,8,9,10, 13, end-5), :]
describe((inputDB_test))[vcat(5,6,8,9,10, 13, end-5), :]
describe((inputDB_pest))[vcat(5,6,8,9,10, 13, end-2), :]

Yy_train = deepcopy(inputDB[:, end-5])
sampleW = []
for w in Vector(Yy_train)
    if w == 0
        push!(sampleW, 0.7183)
    elseif w == 1
        push!(sampleW, 1.6453)
    end
end

Yy_trainWhole = deepcopy(inputDBInputDB_test[:, end-5])
sampleTW = []
for w in Vector(Yy_trainWhole)
    if w == 0
        push!(sampleTW, 0.7183)
    elseif w == 1
        push!(sampleTW, 1.6453)
    end
end
# --------------------------------------------------------------------------------------------------

model = RandomForestClassifier(
      n_estimators = 400, 
      max_depth = 60, 
      min_samples_leaf = 16, 
      min_samples_split = 8, 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42, 
      class_weight= Dict(0=>0.7183, 1=>1.6453)
      )

fit!(model, Matrix(inputDB[:, vcat(5,6,8,9,10, 13, end-5)]), Vector(inputDB[:, end-4]))

# saving model
modelSavePath = "F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5thresholdNms1erFilter_RFwithhlnew.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------

model = RandomForestClassifier(
      n_estimators = 400, 
      max_depth = 30, 
      min_samples_leaf = 8, 
      min_samples_split = 7, 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42, 
      class_weight= Dict(0=>0.7183, 1=>1.6453)
      )

fit!(model, Matrix(inputDB[:, vcat(5,6,8,9,10, 13)]), Vector(inputDB[:, end-4]))

# saving model
modelSavePath = "F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5thresholdNms1erFilter_RFwithhlnew.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------

describe((inputDB))[vcat(5,6,8,9,10, 13, end-5, end-4), :]

#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5thresholdNms1erFilter_RFwithhlnew.joblib")
size(model)
# training performace, withDeltaRi vs. withoutDeltaRi
predictedTPTN_train = predict(model, Matrix(inputDB[:, vcat(5,6,8,9,10, 13, end-5)]))
inputDB[!, "withDeltaRIpredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 475245 x 22+1 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5thresholdNms1erFilter_RFwithhlnew.joblib")
size(model)
predictedTPTN_train = predict(model, Matrix(inputDB[:, vcat(5,6,8,9,10, 13)]))
inputDB[!, "withoutDeltaRIpredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 475245 x 22+1 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB)

# ==================================================================================================
inputDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv", DataFrame)
# RF: 1, 0.17533061894391314, 0.4187249920221065
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])
# RF: 0.17508897273975843
rSquare_train = rSquareDetermination(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])

# 475245 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(5,6,8,9,10, 13, end-6)]))
# RF: 0.7257990739857118
f1_train = f1_score(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])
# RF: 0.5989782032661949
mcc_train = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])

inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 475245 x (23+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB_withDeltaRiTPTN)

describe((inputDB_withDeltaRiTPTN))[end-5:end, :]

# --------------------------------------------------------------------------------------------------
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv", DataFrame)
# RF: 1, 0.17687718966007007, 0.420567699259073
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])
# RF: 0.16511903888127144
rSquare_train = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])

# 475245 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withoutDeltaRiTPTN[:, vcat(5,6,8,9,10, 13)]))
# RF: 0.7172362755651238
f1_train = f1_score(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])
# RF: 0.5892000526505055
mcc_train = matthews_corrcoef(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])

inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 475245 x (23+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB_withoutDeltaRiTPTN)

describe((inputDB_withoutDeltaRiTPTN))[end-5:end, :]

# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5thresholdNms1erFilter_RFwithhlnew.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5thresholdNms1erFilter_RFwithhlnew.joblib")
size(modelRF_TPTN)

# ==================================================================================================

describe((inputDB_test))[end-5:end, :]

predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, vcat(5,6,8,9,10, 13, end-5)]))
inputDB_test[!, "withDeltaRIpredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 119054 x 22+1 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB_test)
# --------------------------------------------------------------------------------------------------
predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, vcat(5,6,8,9,10, 13)]))
inputDB_test[!, "withoutDeltaRIpredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 119054 x 22+1 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withoutDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB_test)

# ==================================================================================================
inputTestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv", DataFrame)
describe((inputTestDB_withDeltaRiTPTN))[end-6:end, :]

# RF: 1, 0.24115947385220152, 0.49107990577114996
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])
# RF: -0.1411513110854168
rSquare_val = rSquareDetermination(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])

# 119054 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputTestDB_withDeltaRiTPTN[:, vcat(5,6,8,9,10, 13, end-6)]))
# RF: 0.613637281156222
f1_test = f1_score(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])
# RF: 0.43904565366065074
mcc_test = matthews_corrcoef(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])

inputTestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputTestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 119054 x 23+2 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withDeltaRiandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputTestDB_withDeltaRiTPTN)

describe((inputTestDB_withDeltaRiTPTN))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
inputTestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withoutDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv", DataFrame)
describe((inputTestDB_withoutDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.27771431451274214, 0.5269860667159447
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputTestDB_withoutDeltaRiTPTN[:, end-5], inputTestDB_withoutDeltaRiTPTN[:, end])
# RF: -0.31646535672567566
rSquare_val = rSquareDetermination(inputTestDB_withoutDeltaRiTPTN[:, end-5], inputTestDB_withoutDeltaRiTPTN[:, end])

# 119054 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputTestDB_withoutDeltaRiTPTN[:, vcat(5,6,8,9,10, 13)]))
# RF: 0.5409765511113579
f1_test = f1_score(inputTestDB_withoutDeltaRiTPTN[:, end-5], inputTestDB_withoutDeltaRiTPTN[:, end])
# RF: 0.34189667190045764
mcc_test = matthews_corrcoef(inputTestDB_withoutDeltaRiTPTN[:, end-5], inputTestDB_withoutDeltaRiTPTN[:, end])

inputTestDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputTestDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 119054 x 23+2 df 
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withoutDeltaRiandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputTestDB_withoutDeltaRiTPTN)

describe((inputTestDB_withoutDeltaRiTPTN))[end-4:end, :]

# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5thresholdNms1erFilter_RFwithhlnew.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5thresholdNms1erFilter_RFwithhlnew.joblib")
size(modelRF_TPTN)

# ==================================================================================================

describe((inputDB_pest))[end-2:end, :]

predictedTPTN_pest = predict(modelRF_TPTN, Matrix(inputDB_pest[:, vcat(5,6,8,9,10, 13, end-2)]))
inputDB_pest[!, "withDeltaRIpredictTPTN"] = predictedTPTN_pest
# save, ouputing testSet df 13278 x 19+1 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB_pest)
# --------------------------------------------------------------------------------------------------
predictedTPTN_pest = predict(modelRF_TPTN, Matrix(inputDB_pest[:, vcat(5,6,8,9,10, 13)]))
inputDB_pest[!, "withoutDeltaRIpredictTPTN"] = predictedTPTN_pest
# save, ouputing testSet df 13278 x 19+1 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputDB_pest)

# ==================================================================================================
inputPestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv", DataFrame)
describe((inputPestDB_withDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.379951799969875, 0.6164023036701558
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])
# RF: -0.3821740704036647
rSquare_val = rSquareDetermination(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])

# 13278 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withDeltaRiTPTN[:, vcat(5,6,8,9,10, 13, end-3)]))
# RF: 0.65560789132364
f1_test = f1_score(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])
# RF: 0.3013579606806383
mcc_test = matthews_corrcoef(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])

inputPestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 13278 x 20+2 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withDeltaRiandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputPestDB_withDeltaRiTPTN)

describe((inputPestDB_withDeltaRiTPTN))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
inputPestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutDeltaRiandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter.csv", DataFrame)
describe((inputPestDB_withoutDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.38439524024702515, 0.6199961614776539
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-2], inputPestDB_withoutDeltaRiTPTN[:, end])
# RF: -0.3131364033713342
rSquare_val = rSquareDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-2], inputPestDB_withoutDeltaRiTPTN[:, end])

# 13278 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withoutDeltaRiTPTN[:, vcat(5,6,8,9,10, 13)]))
# RF: 0.639802399435427
f1_test = f1_score(inputPestDB_withoutDeltaRiTPTN[:, end-2], inputPestDB_withoutDeltaRiTPTN[:, end])
# RF: 0.321333740634701
mcc_test = matthews_corrcoef(inputPestDB_withoutDeltaRiTPTN[:, end-2], inputPestDB_withoutDeltaRiTPTN[:, end])

inputPestDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 13278 x 20+2 df 
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutDeltaRiandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter.csv"
CSV.write(savePath, inputPestDB_withoutDeltaRiTPTN)

describe((inputPestDB_withoutDeltaRiTPTN))[end-4:end, :]
