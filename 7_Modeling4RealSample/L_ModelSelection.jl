VERSION
using Pkg
#Pkg.add("ScikitLearn")
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\T1208\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
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

# inputing 1686319 x 22 df
# 0: 1535009; 1: 151310 = 0.5493; 5.5724
trainDEFSDf = CSV.read("F:\\UvA\\F\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
trainDEFSDf[trainDEFSDf.LABEL .== 1, :]
describe(trainDEFSDf)

Yy_train = deepcopy(trainDEFSDf[:, end-4])  # 0.5493; 5.5724
sampleW = []
for w in Vector(Yy_train)
    if w == 0
        push!(sampleW, 0.5493)
    elseif w == 1
        push!(sampleW, 5.5724)
    end
end

# inputing 421381 x 22 df
# 0: 383416; 1: 37965 = 0.5495; 5.5496
testDEFSDf = CSV.read("F:\\UvA\\F\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
testDEFSDf[testDEFSDf.LABEL .== 1, :]

Yy_val = deepcopy(testDEFSDf[:, end-4])  # 0.5495; 5.5496
sampletestW = []
for w in Vector(Yy_val)
    if w == 0
        push!(sampletestW, 0.5495)
    elseif w == 1
        push!(sampletestW, 5.5496)
    end
end

# 2107700 x 22 df; 
# 1686319+421381= 2107700, 0:1918425; 1:189275 = 
wholeDEFSDf = vcat(trainDEFSDf, testDEFSDf)
sort!(wholeDEFSDf, [:ENTRY])
wholeDEFSDf[wholeDEFSDf.LABEL .== 1, :]


# 10908 x 19 df
# 0: 7173; 1: 3735 = 0.7604; 1.4602
noTeaDEFSDf = CSV.read("F:\\UvA\\F\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
noTeaDEFSDf[noTeaDEFSDf.LABEL .== 1, :]

Yy_test = deepcopy(noTeaDEFSDf[:, end-1])  # 0.7604; 1.4602
samplepestW = []
for w in Vector(Yy_test)
    if w == 0
        push!(samplepestW, 0.7604)
    elseif w == 1
        push!(samplepestW, 1.4602)
    end
end

# 29599 x 19 df
# 1: 8187
TeaDEFSDf = CSV.read("F:\\UvA\\F\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
TeaDEFSDf = TeaDEFSDf[TeaDEFSDf.LABEL .== 1, :]

Yy_test2 = deepcopy(TeaDEFSDf[:, end-1])
samplepest2W = []
for w in Vector(Yy_test2)
    if w == 0
        push!(samplepest2W, 0)
    elseif w == 1
        push!(samplepest2W, 0.5)
    end
end

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: GradientBoostingClassifier
@sk_import linear_model: LogisticRegression
@sk_import ensemble: RandomForestClassifier
@sk_import ensemble: AdaBoostClassifier
@sk_import tree: DecisionTreeClassifier
@sk_import metrics: recall_score
@sk_import neural_network: MLPClassifier
@sk_import svm: SVC
@sk_import neighbors: KNeighborsClassifier
@sk_import inspection: permutation_importance
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

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
describe((trainDEFSDf))[vcat(5,6,7,9,10,13,14, 17), :]
describe((testDEFSDf))[vcat(5,6,9,10,13,14, end-5), :]
describe((noTeaDEFSDf))[vcat(5,6,9,10,13,14, end-2), :]
describe((TeaDEFSDf))[vcat(5,6,9,10,13,14, end-2), :]
describe((TeaDEFSDf))

#===============================================================================#

model = KNeighborsClassifier(
      n_neighbors = 393, 
      weights = "uniform", 
      leaf_size = 300, 
      p = 2, 
      metric = "minkowski"
      )

rank = vcat(5, 7,9,10,13,14, 17)
    N_train = trainDEFSDf
    M_train = vcat(trainDEFSDf, trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
            trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
            trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
            trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
            trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
            trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
            trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
            trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
            trainDEFSDf[trainDEFSDf.LABEL .== 1, :])
    M_val = testDEFSDf
    M_pest = noTeaDEFSDf
    M_pest2 = TeaDEFSDf
    Xx_train = deepcopy(M_train[:, rank])
    nn_train = deepcopy(N_train[:, rank])
    Xx_val = deepcopy(M_val[:, rank])
    Xx_test = deepcopy(M_pest[:, rank])
    Xx_test2 = deepcopy(M_pest2[:, rank])
    #
    Yy_train = deepcopy(M_train[:, end-4])
    mm_train = deepcopy(N_train[:, end-4])
    Yy_val = deepcopy(M_val[:, end-4])
    Yy_test = deepcopy(M_pest[:, end-1])
    Yy_test2 = deepcopy(M_pest2[:, end-1])
## fit ##
    fit!(model, Matrix(Xx_train), Vector(Yy_train))
    importances = permutation_importance(model, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
    print(importances["importances_mean"])
    #"importances_std"  => [0.00390345, 0.00454657, 0.00114062, 0.00124979, 0.0026168, 0.00213004, 0.00140369]
    #"importances_mean" => [0.0160616, 0.0678585, 0.000733407, -0.00270444, 0.00319032, 0.00462046, -0.00619729]

# saving model
modelSavePath = "F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_KNN11_noFilterLog(UsrFragMatchRatio).joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------

model_noRI = KNeighborsClassifier(
      n_neighbors = 393, 
      weights = "uniform", 
      leaf_size = 300, 
      p = 2, 
      metric = "minkowski"
      )

rank2 = vcat(5, 7,9,10,13,14)
    Xx_train_noRI = deepcopy(M_train[:, rank2])
    nn_train_noRI = deepcopy(N_train[:, rank2])
    Xx_val_noRI = deepcopy(M_val[:, rank2])
    Xx_test_noRI = deepcopy(M_pest[:, rank2])
    Xx_test2_noRI = deepcopy(M_pest2[:, rank2])
## fit ##
    fit!(model_noRI, Matrix(Xx_train_noRI), Vector(Yy_train))
    importances2 = permutation_importance(model_noRI, Matrix(Xx_test_noRI), Vector(Yy_test), n_repeats=10, random_state=42)
    print(importances2["importances_mean"])
    #"importances_std"  => [0.0038459, 0.00517923, 0.0017152, 0.00203555, 0.00159498, 0.00266296]
    #"importances_mean" => [0.0126421, 0.0797488, 0.00553722, -0.00176934, 0.00484965, 0.00163183]

# saving model
modelSavePath = "F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_KNN11_noFilterLog(UsrFragMatchRatio)_noRI.joblib"
jl.dump(model_noRI, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------

describe((inputDB_pest))[vcat(collect(5:12), end-1), :]

#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_KNN11_noFilterLog(UsrFragMatchRatio).joblib")
# training performace, withDeltaRi vs. withoutDeltaRi
predictedTPTN_train = predict(model, Matrix(trainDEFSDf[:, rank]))
train_withRI = deepcopy(trainDEFSDf)
train_withRI[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTN_noFilterLog(UsrFragMatchRatio)_KNN.csv"
CSV.write(savePath, train_withRI)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model_noRI = jl.load("F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_KNN11_noFilterLog(UsrFragMatchRatio)_noRI.joblib")
#predictedTPTN_train = predict(model, Matrix(inputDB[:, 5:12]))
predictedTPTN_train = predict(model_noRI, Matrix(trainDEFSDf[:, rank2]))
train_withoutRI = deepcopy(trainDEFSDf)
train_withoutRI[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_TrainDF_withoutDeltaRIandPredictedTPTN_noFilterLog(UsrFragMatchRatio)_KNN.csv"
CSV.write(savePath, train_withoutRI)

# ==================================================================================================
inputDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv", DataFrame)
# 1, 0.10005448199956268, 0.3163138978918926
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
# -0.12879993787163913
rSquare_train = rSquareDetermination(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])

# 3283078 × 2 Matrix
#pTP_train = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]))
pTP_train = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:10), 13, end-4)]))
# 0.6251495964614
f1_train = f1_score(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
# 0.6227021963484577
mcc_train = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])

inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 3283078 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputDB_withDeltaRiTPTN)

describe((inputDB_withDeltaRiTPTN))[end-5:end, :]
# --------------------------------------------------------------------------------------------------
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv", DataFrame)
# 1, 0.12032581701621878, 0.3468801190847045
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])
# -0.3072022185406238
rSquare_train = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])

# 3283078 × 2 Matrix
#pTP_train = predict_proba(model, Matrix(inputDB_withoutDeltaRiTPTN[:, 5:12]))
pTP_train = predict_proba(model, Matrix(inputDB_withoutDeltaRiTPTN[:, vcat(collect(5:10), 13)]))
# 0.576523746673488
f1_train = f1_score(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])
# 0.5739855274485236
mcc_train = matthews_corrcoef(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])

inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 3283078 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputDB_withoutDeltaRiTPTN)

describe((inputDB_withoutDeltaRiTPTN))[end-5:end, :]

# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
size(modelRF_TPTN)

# ==================================================================================================

describe((inputDB_pest))[end-5:end, :]

#predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_pest[:, vcat(collect(5:12), end-1)]))
predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_pest[:, vcat(collect(5:10), 13, end-1)]))
inputDB_pest[!, "withDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 136678 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputDB_pest)
# --------------------------------------------------------------------------------------------------
#predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_pest[:, 5:12]))
predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_pest[:, vcat(collect(5:10), 13)]))
inputDB_pest[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 136678 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputDB_pest)

# ==================================================================================================
inputPestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv", DataFrame)
describe((inputPestDB_withDeltaRiTPTN))[end-5:end, :]

# 1, 0.06034705199329119, 0.24565636973889196
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])
# 0.5066498592682847
rSquare_val = rSquareDetermination(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])

# 136678 × 2 Matrix
#pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-2)]))
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withDeltaRiTPTN[:, vcat(collect(5:10), 13, end-2)]))
# 0.7911830357142857
f1_test = f1_score(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])
# 0.7560033726659677
mcc_test = matthews_corrcoef(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])

inputPestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 136678 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputPestDB_withDeltaRiTPTN)

describe((inputPestDB_withDeltaRiTPTN))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
inputPestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv", DataFrame)
describe((inputPestDB_withoutDeltaRiTPTN))[end-5:end, :]

# 1, 0.06133079602631918, 0.2476505522431137
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])
# 0.4987356913594312
rSquare_val = rSquareDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])

# 136678 × 2 Matrix
#pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withoutDeltaRiTPTN[:, 5:12]))
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withoutDeltaRiTPTN[:, vcat(collect(5:10), 13)]))
# 0.7899707295521069
f1_test = f1_score(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])
# 0.7543562981953349
mcc_test = matthews_corrcoef(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])

inputPestDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 136678 x 19+2 df 
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputPestDB_withoutDeltaRiTPTN)

describe((inputPestDB_withoutDeltaRiTPTN))[end-4:end, :]
