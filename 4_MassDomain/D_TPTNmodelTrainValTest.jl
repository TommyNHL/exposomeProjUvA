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

## Average accuracy
function avgAcc(arrAcc, cv)
    sumAcc = 0
    for acc in arrAcc
        sumAcc += acc
    end
    return sumAcc / cv
end

# modeling, 6 x 11 = 66 times
function optimRandomForestRegressor(inputDB, inputDB_test)
    #leaf_r = [collect(4:2:10);15;20]
    leaf_r = vcat(collect(1:1:6))
    tree_r = vcat(collect(50:50:400),collect(500:100:700))
    #tree_r = collect(50:50:300)
    rs = vcat(1, 42)
    z = zeros(1,9)
    itr = 1
    while itr < 67
        l = rand(leaf_r)
        t = rand(tree_r)
        for s in rs
            println("itr=", itr, ", leaf=", l, ", tree=", t)
            MaxFeat = Int64(9)
            println("## split ##")
            M_train = inputDB
            M_val = inputDB_test
            Xx_train = deepcopy(M_train[:, vcat(collect(5:12), end-3)])
            Yy_train = deepcopy(M_train[:, end-2])
            Xx_val = deepcopy(M_val[:, vcat(collect(5:12), end-3)])
            Yy_val = deepcopy(M_val[:, end-2])
            println("## Regression ##")
            reg = RandomForestClassifier(n_estimators=t, min_samples_leaf=l, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=s, class_weight=Dict(0=>0.529, 1=>9.097))
            println("## fit ##")
            fit!(reg, Matrix(Xx_train), Vector(Yy_train))
            if itr == 1
                z[1,1] = l
                z[1,2] = t
                z[1,3] = score(reg, Matrix(Xx_train), Vector(Yy_train))
                z[1,4] = score(reg, Matrix(df_train[:, vcat(collect(5:12), end-3)]), Vector(df_train[:, end-2]))
                println("## CV ##")
                acc10_train = cross_val_score(reg, Matrix(df_train[:, vcat(collect(5:12), end-3)]), Vector(df_train[:, end-2]); cv = 10)
                z[1,5] = avgAcc(acc10_train, 10)
                z[1,6] = score(reg, Matrix(Xx_val), Vector(Yy_val))
                z[1,7] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                z[1,8] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                z[1,9] = s
                println(z[end, :])
            else
                println("## CV ##")
                itrain= score(reg, Matrix(Xx_train), Vector(Yy_train)) 
                traintrain = score(reg, Matrix(df_train[:, vcat(collect(5:12), end-3)]), Vector(df_train[:, end-2]))
                acc10_train = cross_val_score(reg, Matrix(df_train[:, vcat(collect(5:12), end-3)]), Vector(df_train[:, end-2]); cv = 10)
                traincvtrain = avgAcc(acc10_train, 10) 
                ival = score(reg, Matrix(Xx_val), Vector(Yy_val))
                f1s = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                mccs = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                z = vcat(z, [l t itrain traintrain traincvtrain ival f1s mccs s])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], accuracy_10Ftrain = z[:,3], accuracy_train = z[:,4], avgAccuracy10FCV_train = z[:,5], accuracy_val = z[:,6], f1_val = z[:,7], mcc_val = z[:,8], state = z[:,9])
    z_df_sorted = sort(z_df, [:mcc_val, :f1_val, :accuracy_val, :avgAccuracy10FCV_train, :accuracy_train, :accuracy_10Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestRegressor(inputDB, inputDB_test)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithDeltaRi.csv"
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
      n_estimators = 700, 
      #max_depth = 10, 
      min_samples_leaf = 2, 
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
modelSavePath = "F:\\modelTPTNModeling_withDeltaRi.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\modelTPTNModeling_withoutDeltaRi.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\modelTPTNModeling_WholeWithDeltaRi.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\modelTPTNModeling_WholeWithoutDeltaRi.joblib"
jl.dump(model, modelSavePath, compress = 5)

# training performace, withDeltaRi vs. withoutDeltaRi
predictedTPTN_train = predict(model,  Matrix(inputDB[:, vcat(collect(5:12), end-3)]))
inputDB[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
predictedTPTN_train = predict(model, Matrix(inputDB[:, 5:12]))
inputDB[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
predictedTPTN_train = predict(model,  Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]))
inputDBInputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 4103848 x 19 df
savePath = "F:\\dataframeTPTNModeling_WholeDF_withDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDBInputDB_test)
# --------------------------------------------------------------------------------------------------
predictedTPTN_train = predict(model, Matrix(inputDBInputDB_test[:, 5:12]))
inputDBInputDB_test[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 4103848 x 19 df
savePath = "F:\\dataframeTPTNModeling_WholeDF_withoutDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDBInputDB_test)

# ==================================================================================================

# 1.0, 0.0526891779639916, 0.2295412336901403
maxAE_train, MSE_train, RMSE_train = errorDetermination(Matrix(inputDB[:, vcat(collect(5:12), end-4)]), predictedTPTN_train)
# -3.158185021345462
rSquare_train = rSquareDetermination(Matrix(inputDB[:, vcat(collect(5:12), end-4)]), predictedTPTN_train)
## accuracy, 0.9977639885497694
acc1_train = score(model, Matrix(inputDB[:, vcat(collect(5:12), end-4)]), Vector(inputDB[:, end-3]))
# 0.938269856354399, 0.9336424942432107, 0.9370105577882192
acc5_train = cross_val_score(model, Matrix(inputDB[:, vcat(collect(5:12), end-4)]), Vector(inputDB[:, end-3]); cv = 3)
# 0.9363076361286096
avgAcc_train = avgAcc(acc5_train, 3)
# 3283078 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB[:, vcat(collect(5:12), end-4)]))
# 0.9800508172887482
f1_train = f1_score(Vector(inputDB[:, end-3]), predictedTPTN_train)
# 0.979071377443971
mcc_train = matthews_corrcoef(Vector(inputDB[:, end-3]), predictedTPTN_train)

inputDB[!, "pTP_train1"] = pTP_train[:, 1]
inputDB[!, "pTP_train2"] = pTP_train[:, 2]
# save, ouputing trainSet df 3283078 x (19+1+2)
savePath = "F:\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB)

describe((inputDB))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
# 1.0, 0.10082538783857325, 0.3175301368981742
maxAE_train, MSE_train, RMSE_train = errorDetermination(Matrix(inputDB[:, 5:12]), predictedTPTN_train)
# -6.336295384890479
rSquare_train = rSquareDetermination(Matrix(inputDB[:, 5:12]), predictedTPTN_train)
## accuracy, 0.9484736579514712
acc1_train = score(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-3]))
#  0.9096458203881721, 0.900339010928762, 0.9056040162360032
acc5_train = cross_val_score(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-3]); cv = 3)
# 0.9051962825176457
avgAcc_train = avgAcc(acc5_train, 3)
# 3283078 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB[:, 5:12]))
# 0.6807213566487681
f1_train = f1_score(Vector(inputDB[:, end-3]), predictedTPTN_train)
# 0.6983353384554911
mcc_train = matthews_corrcoef(Vector(inputDB[:, end-3]), predictedTPTN_train)

inputDB[!, "pTP_train1"] = pTP_train[:, 1]
inputDB[!, "pTP_train2"] = pTP_train[:, 2]
# save, ouputing trainSet df 3283078 x (19+1+2)
savePath = "F:\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB)

describe((inputDB))[end-4:end, :]

# --------------------------------------------------------------------------------------------------
# 1.0, 0.05271413799292683, 0.2295955966322674
maxAE_train, MSE_train, RMSE_train = errorDetermination(Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-4)]), predictedTPTN_train)
# -3.1641988462358626
rSquare_train = rSquareDetermination(Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-4)]), predictedTPTN_train)
## accuracy, 0.9978222877650439
acc1_train = score(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-4)]), Vector(inputDBInputDB_test[:, end-3]))
#  0.9381578274059724, 0.9333338206805805, 0.9364983171874954
acc5_train = cross_val_score(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-4)]), Vector(inputDBInputDB_test[:, end-3]); cv = 3)
# 0.9359966550913494
avgAcc_train = avgAcc(acc5_train, 3)
# 4103848 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-4)]))
# 0.9805646832654464
f1_train = f1_score(Vector(inputDBInputDB_test[:, end-3]), predictedTPTN_train)
# 0.979607915383195
mcc_train = matthews_corrcoef(Vector(inputDBInputDB_test[:, end-3]), predictedTPTN_train)

inputDBInputDB_test[!, "pTP_train1"] = pTP_train[:, 1]
inputDBInputDB_test[!, "pTP_train2"] = pTP_train[:, 2]
# save, ouputing trainSet df 4103848 x (19+1+2)
savePath = "F:\\dataframeTPTNModeling_WholeDF_withDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDBInputDB_test)

describe((inputDBInputDB_test))[end-4:end, :]

# --------------------------------------------------------------------------------------------------
# 1.0, 0.10103712953381895, 0.3178633818699772
maxAE_train, MSE_train, RMSE_train = errorDetermination(Matrix(inputDBInputDB_test[:, 5:12]), predictedTPTN_train)
# -6.348633334302107
rSquare_train = rSquareDetermination(Matrix(inputDBInputDB_test[:, 5:12]), predictedTPTN_train)
## accuracy, 0.9482556371483544
acc1_train = score(model, Matrix(inputDBInputDB_test[:, 5:12]), Vector(inputDBInputDB_test[:, end-3]))
# 0.9096648269308089, 0.9002690156803976, 0.9053231555585446
acc5_train = cross_val_score(model, Matrix(inputDBInputDB_test[:, 5:12]), Vector(inputDBInputDB_test[:, end-3]); cv = 3)
# 0.9050856660565838
avgAcc_train = avgAcc(acc5_train, 3)
# 4103848 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDBInputDB_test[:, 5:12]))
# 0.6798282977077713
f1_train = f1_score(Vector(inputDBInputDB_test[:, end-3]), predictedTPTN_train)
# 0.6975629220701743
mcc_train = matthews_corrcoef(Vector(inputDBInputDB_test[:, end-3]), predictedTPTN_train)

inputDBInputDB_test[!, "pTP_train1"] = pTP_train[:, 1]
inputDBInputDB_test[!, "pTP_train2"] = pTP_train[:, 2]
# save, ouputing trainSet df 4103848 x (19+1+2)
savePath = "F:\\dataframeTPTNModeling_WholeDF_withoutDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDBInputDB_test)

describe((inputDBInputDB_test))[end-4:end, :]


# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\modelTPTNModeling_withDeltaRi.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\modelTPTNModeling_withoutDeltaRi.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\modelTPTNModeling_WholeWithDeltaRi.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\modelTPTNModeling_WholeWithoutDeltaRi.joblib")
size(modelRF_TPTN)

# ==================================================================================================

predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-3)]))
inputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 820770 x 19 df
savePath = "F:\\dataframeTPTNModeling_TestDF_withDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB_test)
# --------------------------------------------------------------------------------------------------
predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]))
inputDB_test[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 820770 x 19 df
savePath = "F:\\dataframeTPTNModeling_TestDF_withoutDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB_test)
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------


# ==================================================================================================
# 1.0, 0.050030569649485414, 0.223675143119403
maxAE_val, MSE_val, RMSE_val = errorDetermination(Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]), predictedTPTN_test)
# -2.937069985135655
rSquare_val = rSquareDetermination(Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]), predictedTPTN_test)
## accuracy, 0.983558122251057
acc1_val = score(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]), Vector(inputDB_test[:, end-3]))
# 0.9394166453452246, 0.9342994992507037, 0.9364669761321686
acc5_val = cross_val_score(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]), Vector(inputDB_test[:, end-3]); cv = 3)
# 0.9367277069093656
avgAcc_val = avgAcc(acc5_val, 3)
# 820770 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]))
# 0.8487802691587948
f1_test = f1_score(Vector(inputDB_test[:, end-3]), predictedTPTN_test)
# 0.8401489159641009
mcc_test = matthews_corrcoef(Vector(inputDB_test[:, end-3]), predictedTPTN_test)

inputDB_test[!, "pTP_test1"] = pTP_test[:, 1]
inputDB_test[!, "pTP_test2"] = pTP_test[:, 2]
# save, ouputing trainSet df 820770 x (19+1+2)
savePath = "F:\\dataframeTPTNModeling_TestDF_withDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB_test)

describe((inputDB_test))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
# 1.0, 0.09482519232746037, 0.3079369940872002
maxAE_val, MSE_val, RMSE_val = errorDetermination(Matrix(inputDB_test[:, 5:12]), predictedTPTN_test)
# -6.173915730914463
rSquare_val = rSquareDetermination(Matrix(inputDB_test[:, 5:12]), predictedTPTN_test)
## accuracy, 0.9167038268942578
acc1_val = score(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]), Vector(inputDB_test[:, end-3]))
# 0.9110091743119266, 0.9014693519499982, 0.9068094594100662
acc5_val = cross_val_score(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]), Vector(inputDB_test[:, end-3]); cv = 3)
# 0.9064293285573303
avgAcc_val = avgAcc(acc5_val, 3)
# 820770 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]))
# 0.45913467243657197
f1_test = f1_score(Vector(inputDB_test[:, end-3]), predictedTPTN_test)
# 0.43929652841599554
mcc_test = matthews_corrcoef(Vector(inputDB_test[:, end-3]), predictedTPTN_test)

inputDB_test[!, "pTP_test1"] = pTP_test[:, 1]
inputDB_test[!, "pTP_test2"] = pTP_test[:, 2]
# save, ouputing trainSet df 820770 x 19+2 df 
savePath = "F:\\dataframeTPTNModeling_TestDF_withoutDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB_test)

describe((inputDB_test))[end-4:end, :]
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
