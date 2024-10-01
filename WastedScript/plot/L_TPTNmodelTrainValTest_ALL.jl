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

# inputing 3391333 x 4+8+1+2+1+1+2+2 df
inputDB = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainYesDFwithhl_all.csv", DataFrame)
sort!(inputDB, [:ENTRY])
insertcols!(inputDB, 10, ("MatchRatio"=>float(0)))
inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
describe(inputDB[inputDB.LABEL .== 1, :])
inputDB = inputDB[inputDB.MS1Error .>= float(-0.001), :]
inputDB = inputDB[inputDB.MS1Error .<= float(0.001), :]
for i = 1:size(inputDB, 1)
    inputDB[i, "RefMatchFragRatio"] = log10(inputDB[i, "RefMatchFragRatio"])
    inputDB[i, "UsrMatchFragRatio"] = log10(inputDB[i, "UsrMatchFragRatio"])
    inputDB[i, "FinalScoreRatio"] = log10(inputDB[i, "FinalScoreRatio"])
    inputDB[i, "MatchRatio"] = inputDB[i, "DirectMatch"] - inputDB[i, "ReversMatch"]
end
# save, ouputing 485631 x 21+1 df, 0:334321; 1:151310 = 0.7263; 1.6048
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl0d5FinalScoreRatio2DE2Filter_all.csv"
CSV.write(savePath, inputDB)
inputDB[inputDB.LABEL .== 1, :]

describe((inputDB_test))[1:14, :]
# inputing 847838 x 4+8+1+2+1+1+2+2 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestYesDFwithhl_all.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
insertcols!(inputDB_test, 10, ("MatchRatio"=>float(0)))
inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
describe(inputDB_test[inputDB_test.LABEL .== 1, :])
inputDB_test = inputDB_test[inputDB_test.MS1Error .>= float(-0.001), :]
inputDB_test = inputDB_test[inputDB_test.MS1Error .<= float(0.001), :]
for i = 1:size(inputDB_test, 1)
    inputDB_test[i, "RefMatchFragRatio"] = log10(inputDB_test[i, "RefMatchFragRatio"])
    inputDB_test[i, "UsrMatchFragRatio"] = log10(inputDB_test[i, "UsrMatchFragRatio"])
    inputDB_test[i, "FinalScoreRatio"] = log10(inputDB_test[i, "FinalScoreRatio"])
    inputDB_test[i, "MatchRatio"] = inputDB_test[i, "DirectMatch"] - inputDB_test[i, "ReversMatch"]
end
# save, ouputing 121946 x 21+1 df, 0:83981; 1:37965 = 0.7260; 1.6060
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDFwithhl0d5FinalScoreRatio2DE2Filter_all.csv"
CSV.write(savePath, inputDB_test)
inputDB_test[inputDB_test.LABEL .== 1, :]

# 607577 x 22 df; 
# 485631+121946= 607577, 0:418302; 1:189275 = 
inputDBInputDB_test = vcat(inputDB, inputDB_test)
sort!(inputDBInputDB_test, [:ENTRY])
inputDBInputDB_test[inputDBInputDB_test.LABEL .== 1, :]


#TP/TN prediction
inputDB_pest2 = CSV.read("F:\\UvA\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
sort!(inputDB_pest2, [:ENTRY])
insertcols!(inputDB_pest2, 10, ("MatchDiff"=>float(0)))
inputDB_pest2 = inputDB_pest2[inputDB_pest2.FinalScoreRatio .>= float(0.5), :]
inputDB_pest2 = inputDB_pest2[inputDB_pest2.Leverage .<= 0.14604417882015916, :]
describe(inputDB_pest2[inputDB_pest2.LABEL .== 0, :])
describe(inputDB_pest2[inputDB_pest2.LABEL .== 1, :])
inputDB_pest2 = inputDB_pest2[inputDB_pest2.MS1Error .>= float(-0.061), :]
inputDB_pest2 = inputDB_pest2[inputDB_pest2.MS1Error .<= float(0.058), :]
for i = 1:size(inputDB_pest2, 1)
    inputDB_pest2[i, "RefMatchFragRatio"] = log10(inputDB_pest2[i, "RefMatchFragRatio"])
    inputDB_pest2[i, "UsrMatchFragRatio"] = log10(inputDB_pest2[i, "UsrMatchFragRatio"])
    inputDB_pest2[i, "FinalScoreRatio"] = log10(inputDB_pest2[i, "FinalScoreRatio"])
    inputDB_pest2[i, "MatchDiff"] = inputDB_pest2[i, "DirectMatch"] - inputDB_pest2[i, "ReversMatch"]
end
# save, ouputing 4757 x 18+1 df, 0:3423; 1:1334 = 0.6949; 1.7830
savePath = "F:\\UvA\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_dataframeTPTNModeling_testDFwithhl0d5FinalScoreRatioDEFilter.csv"
CSV.write(savePath, inputDB_pest2)
inputDB_pest2[inputDB_pest2.LABEL .== 1, :]

Yy_train = deepcopy(inputDB[:, end-4])  # 0.7263; 1.6048
sampleW = []
for w in Vector(Yy_train)
    if w == 0
        push!(sampleW, 0.7263)
    elseif w == 1
        push!(sampleW, 1.6048)
    end
end 

Yy_val = deepcopy(inputDB_test[:, end-4])  # 0.7260; 1.6060
sampletestW = []
for w in Vector(Yy_val)
    if w == 0
        push!(sampletestW, 0.7260)
    elseif w == 1
        push!(sampletestW, 1.6060)
    end
end 


Yy_test2 = deepcopy(inputDB_pest2[:, end-1])  # 0.6949; 1.7830
samplepest2W = []
for w in Vector(Yy_test2)
    if w == 0
        push!(samplepest2W, 0.6949)
    elseif w == 1
        push!(samplepest2W, 1.7830)
    end
end 

f1 = make_scorer(f1_score, pos_label=1, average="binary")

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: GradientBoostingClassifier
@sk_import linear_model: LogisticRegression
@sk_import ensemble: RandomForestClassifier
@sk_import ensemble: AdaBoostClassifier
@sk_import tree: DecisionTreeClassifier
@sk_import metrics: recall_score
@sk_import neural_network : MLPClassifier
@sk_import svm : SVC
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
describe((inputDB))[vcat(5,6,8,9,10, 13, end-5), :]
describe((inputDB_test))[vcat(5,6,8,9,10, 13, end-5), :]
describe((inputDB_pest2))[vcat(5,6,8,9,10, 13, end-2), :]
# --------------------------------------------------------------------------------------------------

model = DecisionTreeClassifier(
        max_depth = 27, 
        min_samples_leaf = 153, 
        min_samples_split = 2, 
        random_state = 42, 
        class_weight = Dict(0=>0.7263, 1=>1.6048)
        )  # 0.7263; 1.6048

fit!(model, Matrix(inputDB[:, vcat(5,6,8,9,10, 13, end-5)]), Vector(inputDB[:, end-4]))

# saving model
modelSavePath = "F:\\UvA\\modelTPTNModeling_withDeltaRI_0d5thresholdNms1erFilter_DTwithhlnew.joblib"
jl.dump(model, modelSavePath, compress = 5)

# ==================================================================================================

describe((inputDB))[vcat(5,6,8,9,10, 13, end-5, end-4), :]

#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withDeltaRI_0d5thresholdNms1erFilter_DTwithhlnew.joblib")
# training performace, withDeltaRi vs. withoutDeltaRi
predictedTPTN_train = predict(model, Matrix(inputDB[:, vcat(5,6,8,9,10, 13, end-5)]))
inputDB[!, "withDeltaRIpredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 485631 x 22+1 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDFall_withDeltaRIandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB)

# ==================================================================================================

inputDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDFall_withDeltaRIandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)
# DT: 1, 0.3107173965418188, 0.5574203051036254
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])
# DT: -0.352617425206323
rSquare_train = rSquareDetermination(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])

# 485631 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(5,6,8,9,10, 13, end-6)]))
# DT: 0.5837774320611698, 0.6942769012133072
f1_train = f1_score(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end], sample_weight=sampleW)
# DT: 0.35881099250485715, 0.3841185983978827
mcc_train = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end], sample_weight=sampleW)

inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 485631 x (23+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDFall_withDeltaRIandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB_withDeltaRiTPTN)

describe((inputDB_withDeltaRiTPTN))[end-5:end, :]

# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withDeltaRI_0d5thresholdNms1erFilter_DTwithhlnew.joblib")

# ==================================================================================================

describe((inputDB_test))[end-5:end, :]

predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, vcat(5,6,8,9,10, 13, end-5)]))
inputDB_test[!, "withDeltaRIpredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 121946 x 22+1 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDFall_withDeltaRIandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB_test)

# ==================================================================================================

inputTestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDFall_withDeltaRIandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)
describe((inputTestDB_withDeltaRiTPTN))[end-6:end, :]

# DT: 1, 0.32691519197021635, 0.5717649796640367
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])
# DT: -0.42316117628198713
rSquare_val = rSquareDetermination(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])

# 121946 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputTestDB_withDeltaRiTPTN[:, vcat(5,6,8,9,10, 13, end-6)]))
# DT: 0.5619987255268188, 0.6733883547565545
f1_test = f1_score(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end], sample_weight=sampletestW)
# DT: 0.32361815497352586, 0.3464920995991211
mcc_test = matthews_corrcoef(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end], sample_weight=sampletestW)

inputTestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputTestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 121946 x 23+2 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDFall_withDeltaRIandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputTestDB_withDeltaRiTPTN)

describe((inputTestDB_withDeltaRiTPTN))[end-4:end, :]

# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withDeltaRI_0d5thresholdNms1erFilter_DTwithhlnew.joblib")

# ==================================================================================================

describe((inputDB_pest2))[end-2:end, :]

predictedTPTN_pest = predict(modelRF_TPTN, Matrix(inputDB_pest2[:, vcat(5,6,8,9,10, 13, end-2)]))
inputDB_pest2[!, "withDeltaRIpredictTPTN"] = predictedTPTN_pest
# save, ouputing testSet df 4757 x 19+1 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDFreal_withDeltaRIandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputDB_pest2)

# ==================================================================================================

inputPestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDFreal_withDeltaRIandPredictedTPTN_withhl0d5FinalScoreRatio2DE2Filter_DT.csv", DataFrame)
describe((inputPestDB_withDeltaRiTPTN))[end-5:end, :]

# DT: 1, 0.43178473828042885, 0.657103293463386
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])
# DT: -0.865140506715314
rSquare_val = rSquareDetermination(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])

# 4757 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withDeltaRiTPTN[:, vcat(5,6,8,9,10, 13, end-3)]))
# DT: 0.4111238532110092, 0.5492040894474248
f1_test = f1_score(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end], sample_weight=samplepest2W)
# DT: 0.10619454670087872, 0.11778156982480285
mcc_test = matthews_corrcoef(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end], sample_weight=samplepest2W)

inputPestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 4757 x 20+2 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDFreal_withDeltaRIandPredictedTPTNandpTP_withhl0d5FinalScoreRatio2DE2Filter_DT.csv"
CSV.write(savePath, inputPestDB_withDeltaRiTPTN)

describe((inputPestDB_withDeltaRiTPTN))[end-4:end, :]
