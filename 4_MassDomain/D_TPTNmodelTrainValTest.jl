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
@sk_import ensemble: RandomForestClassifier
@sk_import metrics: recall_score
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

# inputing 820770 x 4+8+1+2+1+1+2 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
# inputing 3283078 x 4+8+1+2+1+1+2 df
inputDB = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF.csv", DataFrame)
sort!(inputDB, [:ENTRY])
# 4103848 x 19 df
inputDBInputDB_test = vcat(inputDB, inputDB_test)
sort!(inputDBInputDB_test, [:ENTRY])
# 136678 x 17 df
inputDB_pest = CSV.read("F:\\UvA\\dataframeTPTNModeling_pest.csv", DataFrame)
sort!(inputDB_pest, [:ENTRY])


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

# modeling, 3 x 10 = 30 times
function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)
    #leaf_r = vcat(collect(6:2:10), collect(12:4:20))
    #leaf_r = vcat(collect(10:1:15))
    leaf_r = vcat(collect(14:1:16))
    #tree_r = vcat(collect(50:50:400), collect(500:100:800))
    tree_r = vcat(collect(50:50:500))
    rs = vcat(1, 42)
    z = zeros(1,11)
    itr = 1
    while itr < 13
        l = rand(leaf_r)
        t = rand(tree_r)
        for s in rs
            println("itr=", itr, ", leaf=", l, ", tree=", t)
            MaxFeat = Int64(9)
            println("## loading in data ##")
            M_train = inputDB
            M_val = inputDB_test
            M_pest = inputDB_pest
            Xx_train = deepcopy(M_train[:, vcat(collect(5:12), end-3)])
            Yy_train = deepcopy(M_train[:, end-2])
            Xx_val = deepcopy(M_val[:, vcat(collect(5:12), end-3)])
            Yy_val = deepcopy(M_val[:, end-2])
            Xx_test = deepcopy(M_pest[:, vcat(collect(5:12), end-1)])
            Yy_test = deepcopy(M_pest[:, end])
            println("## Classification ##")
            reg = RandomForestClassifier(n_estimators=t, min_samples_leaf=l, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=s, class_weight=Dict(0=>0.529, 1=>9.097))
            println("## fit ##")
            fit!(reg, Matrix(Xx_train), Vector(Yy_train))
            if itr == 1
                z[1,1] = l
                z[1,2] = t
                z[1,3] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                z[1,4] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                z[1,5] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                z[1,6] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                z[1,7] = avgScore(f1_10_train, 5)
                z[1,8] = score(reg, Matrix(Xx_test), Vector(Yy_test))
                z[1,9] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z[1,10] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z[1,11] = s
                println(z[end, :])
            else
                itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                traincvtrain = avgScore(f1_10_train, 5) 
                itest = score(reg, Matrix(Xx_test), Vector(Yy_test))
                f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z = vcat(z, [l t itrain jtrain ival jval traincvtrain itest f1s mccs s])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], f1_train = z[:,3], mcc_train = z[:,4], f1_val = z[:,5], mcc_val = z[:,6], f1_5Ftrain = z[:,7], acc_pest = z[:,8], f1_pest = z[:,9], mcc_pest = z[:,10], state = z[:,11])
    z_df_sorted = sort(z_df, [:mcc_pest, :f1_pest, :f1_5Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithDeltaRi5F.csv"
CSV.write(savePath, optiSearch_df)

#= model = RandomForestRegressor()
param_dist = Dict(
      "n_estimators" => 50:50:300, 
      #"max_depth" => 2:2:10, 
      "min_samples_leaf" => 8:8:32, 
      "max_features" => [Int64(ceil((size(x_train,2)-1)/3))], 
      "n_jobs" => [-1], 
      "oob_score" => [true], 
      "random_state" => [1]
      )
gridsearch = GridSearchCV(model, param_dist)
@time fit!(gridsearch, Matrix(x_train), Vector(y_train))
println("Best parameters: $(gridsearch.best_params_)") =#

model = RandomForestClassifier(
      n_estimators = 450, 
      #max_depth = 10, 
      min_samples_leaf = 15, 
      max_features = Int64(9), 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42, 
      class_weight= Dict(0=>0.529, 1=>9.097)
      )

fit!(model, Matrix(inputDB[:, vcat(collect(5:12), end-3)]), Vector(inputDB[:, end-2]))
# --------------------------------------------------------------------------------------------------
fit!(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-2]))
# --------------------------------------------------------------------------------------------------
fit!(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]), Vector(inputDBInputDB_test[:, end-2]))
# --------------------------------------------------------------------------------------------------
fit!(model, Matrix(inputDBInputDB_test[:, 5:12]), Vector(inputDBInputDB_test[:, end-2]))

# saving model
modelSavePath = "F:\\UvA\\modelTPTNModeling_withDeltaRi.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_withoutDeltaRi.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithDeltaRi.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithoutDeltaRi.joblib"
jl.dump(model, modelSavePath, compress = 5)

#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withDeltaRi.joblib")
size(model)
# training performace, withDeltaRi vs. withoutDeltaRi
predictedTPTN_train = predict(model,  Matrix(inputDB[:, vcat(collect(5:12), end-3)]))
inputDB[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withoutDeltaRi.joblib")
size(model)
predictedTPTN_train = predict(model, Matrix(inputDB[:, 5:12]))
inputDB[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithDeltaRi.joblib")
size(model)
predictedTPTN_train = predict(model,  Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]))
inputDBInputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 4103848 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDBInputDB_test)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutDeltaRi.joblib")
size(model)
predictedTPTN_train = predict(model, Matrix(inputDBInputDB_test[:, 5:12]))
inputDBInputDB_test[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 4103848 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDBInputDB_test)

# ==================================================================================================
inputDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTN.csv", DataFrame)
# 1.0, 0.05078740133496676, 0.2253606028900499
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
# 0.06811669590416691
rSquare_train = rSquareDetermination(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
#= ## accuracy, 0.9977639885497694
acc1_train = score(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]), Vector(inputDB_withDeltaRiTPTN[:, end-3]))
# 0.938269856354399, 0.9336424942432107, 0.9370105577882192
acc5_train = cross_val_score(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]), Vector(inputDB_withDeltaRiTPTN[:, end-3]); cv = 3)
# 0.9363076361286096
avgAcc_train = avgAcc(acc5_train, 3) =#
# 3283078 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]))
# 0.6835861964267077
f1_train = f1_score(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
# 0.7005990247521975
mcc_train = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])

inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 3283078 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB_withDeltaRiTPTN)

describe((inputDB_withDeltaRiTPTN))[end-5:end, :]
# --------------------------------------------------------------------------------------------------
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTN.csv", DataFrame)
# 1, 0.12646120500335356, 0.3556138425361892
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])
# -0.8622447329405676
rSquare_train = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])

# 3283078 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withoutDeltaRiTPTN[:, 5:12]))
# 0.464792327326166
f1_train = f1_score(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])
# 0.5118613741680419
mcc_train = matthews_corrcoef(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])

inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 3283078 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB_withoutDeltaRiTPTN)

describe((inputDB_withoutDeltaRiTPTN))[end-5:end, :]

# --------------------------------------------------------------------------------------------------
inputWholeDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withDeltaRiandPredictedTPTN.csv", DataFrame)
# 1, 0.04544661498184143, 0.21318211693723615
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])
# 0.1582952037998726
rSquare_train = rSquareDetermination(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])

# 4103848 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputWholeDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]))
# 0.707165960119328
f1_train = f1_score(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])
# 0.7212525691969476
mcc_train = matthews_corrcoef(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])

inputWholeDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputWholeDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 4103848 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputWholeDB_withDeltaRiTPTN)

describe((inputWholeDB_withDeltaRiTPTN))[end-5:end, :]

# --------------------------------------------------------------------------------------------------
inputWholeDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutDeltaRiandPredictedTPTN.csv", DataFrame)
# 1, 0.12467445188028406, 0.35309269587501246
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])
# -0.8480738936391097
rSquare_train = rSquareDetermination(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])

# 4103848 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputWholeDB_withoutDeltaRiTPTN[:, 5:12]))
# 0.46833847525487066
f1_train = f1_score(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])
# 0.5149516079309533
mcc_train = matthews_corrcoef(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])

inputWholeDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputWholeDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 4103848 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputWholeDB_withoutDeltaRiTPTN)

describe((inputWholeDB_withoutDeltaRiTPTN))[end-5:end, :]


# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withDeltaRi.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutDeltaRi.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithDeltaRi.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutDeltaRi.joblib")
size(modelRF_TPTN)

# ==================================================================================================

describe((inputDB_pest))[end-5:end, :]

predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_pest[:, vcat(collect(5:12), end-1)]))
inputDB_pest[!, "withDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 136678 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB_pest)
# --------------------------------------------------------------------------------------------------
predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_pest[:, 5:12]))
inputDB_pest[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 136678 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB_pest)

# ==================================================================================================
inputPestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withDeltaRiandPredictedTPTN.csv", DataFrame)
describe((inputPestDB_withDeltaRiTPTN))[end-5:end, :]

# 1, 0.05357116726905574, 0.2314544604648088
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])
# 0.283526929864932
rSquare_val = rSquareDetermination(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])

# 136678 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-2)]))
# 0.6718946047678795
f1_test = f1_score(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])
# 0.6427315488312465
mcc_test = matthews_corrcoef(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])

inputPestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 136678 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputPestDB_withDeltaRiTPTN)

describe((inputPestDB_withDeltaRiTPTN))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
inputPestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutDeltaRiandPredictedTPTN.csv", DataFrame)
describe((inputPestDB_withoutDeltaRiTPTN))[end-5:end, :]

# 1, 0.05104698634747362, 0.2259358013849811
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])
# 0.33156593391445466
rSquare_val = rSquareDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])

# 136678 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withoutDeltaRiTPTN[:, 5:12]))
# 0.748240897773608
f1_test = f1_score(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])
# 0.7388999236740152
mcc_test = matthews_corrcoef(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])

inputPestDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 136678 x 19+2 df 
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputPestDB_withoutDeltaRiTPTN)

describe((inputPestDB_withoutDeltaRiTPTN))[end-4:end, :]
