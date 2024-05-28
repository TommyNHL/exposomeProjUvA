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

describe((inputDB_test))[12:14, :]
# inputing 820770 x 4+8+1+2+1+1+2 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
insertcols!(inputDB_test, 10, ("MatchRatio"=>float(0)))
inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB_test, 1)
    inputDB_test[i, "RefMatchFragRatio"] = log10(inputDB_test[i, "RefMatchFragRatio"])
    inputDB_test[i, "UsrMatchFragRatio"] = log10(inputDB_test[i, "UsrMatchFragRatio"])
    inputDB_test[i, "FinalScoreRatio"] = log10(inputDB_test[i, "FinalScoreRatio"])
    inputDB_test[i, "MatchRatio"] = inputDB_test[i, "DirectMatch"] - inputDB_test[i, "ReversMatch"]
    #inputDB_test[i, "MS1Error"] = abs(inputDB_test[i, "MS1Error"])
    #inputDB_test[i, "DeltaRi"] = abs(inputDB_test[i, "DeltaRi"])
end
#inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
# save, ouputing 409126 x 19 df, 0:373107; 1:36019 = 10.3586:1
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_test)
inputDB_test[inputDB_test.LABEL .== 1, :]

# inputing 3283078 x 4+8+1+2+1+1+2 df
inputDB = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF.csv", DataFrame)
sort!(inputDB, [:ENTRY])
insertcols!(inputDB, 10, ("MatchRatio"=>float(0)))
inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB, 1)
    inputDB[i, "RefMatchFragRatio"] = log10(inputDB[i, "RefMatchFragRatio"])
    inputDB[i, "UsrMatchFragRatio"] = log10(inputDB[i, "UsrMatchFragRatio"])
    inputDB[i, "FinalScoreRatio"] = log10(inputDB[i, "FinalScoreRatio"])
    inputDB[i, "MatchRatio"] = inputDB[i, "DirectMatch"] - inputDB[i, "ReversMatch"]
    #inputDB[i, "MS1Error"] = abs(inputDB[i, "MS1Error"])
    #inputDB[i, "DeltaRi"] = abs(inputDB[i, "DeltaRi"])
end
#inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
# save, ouputing 1637238 x 19 df, 0:1492750; 1:144488 = 10.3313:1
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB)
inputDB[inputDB.LABEL .== 1, :]

# 4103848 x 19 df; 
# 409126+1637238= 2046364, 0:1865857; 1:180507 = 10.3368:1
inputDBInputDB_test = vcat(inputDB, inputDB_test)
sort!(inputDBInputDB_test, [:ENTRY])
inputDBInputDB_test[inputDBInputDB_test.LABEL .== 1, :]

# 136678 x 17 df
inputDB_pest = CSV.read("F:\\UvA\\dataframeTPTNModeling_pest.csv", DataFrame)
sort!(inputDB_pest, [:ENTRY])
insertcols!(inputDB_pest, 10, ("MatchRatio"=>float(0)))
inputDB_pest = inputDB_pest[inputDB_pest.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB_pest, 1)
    inputDB_pest[i, "RefMatchFragRatio"] = log10(inputDB_pest[i, "RefMatchFragRatio"])
    inputDB_pest[i, "UsrMatchFragRatio"] = log10(inputDB_pest[i, "UsrMatchFragRatio"])
    inputDB_pest[i, "FinalScoreRatio"] = log10(inputDB_pest[i, "FinalScoreRatio"])
    inputDB_pest[i, "MatchRatio"] = inputDB_pest[i, "DirectMatch"] - inputDB_pest[i, "ReversMatch"]
    #inputDB_pest[i, "MS1Error"] = abs(inputDB_pest[i, "MS1Error"])
    #inputDB_pest[i, "DeltaRi"] = abs(inputDB_pest[i, "DeltaRi"])
end
#inputDB_pest = inputDB_pest[inputDB_pest.FinalScoreRatio .>= float(0.5), :]
# save, ouputing 62008 x 17 df, 0:53162; 1:8846 = 6.0097:1
savePath = "F:\\UvA\\dataframeTPTNModeling_pestDF0d5FinalScoreRatio.csv"
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
#= function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)
    #leaf_r = vcat(collect(6:2:10), collect(12:4:20))
    #leaf_r = vcat(collect(10:1:15))
    #leaf_r = vcat(collect(14:1:16))
    #leaf_r = vcat(1, collect(2:2:8))
    #leaf_r = vcat(collect(6:1:8))
    leaf_r = vcat(6)
    #tree_r = vcat(collect(50:50:400), collect(500:100:800))
    #tree_r = vcat(collect(200:200:2000))
    tree_r = vcat(collect(250:25:300))
    #depth_r = vcat(collect(5:5:30), collect(40:10:70))
    #depth_r = vcat(collect(55:5:75))
    depth_r = vcat(60)
    #split_r = vcat(collect(4:1:6))
    split_r = vcat(4)
    #rs = vcat(1, 42)
    rs = vcat(42)
    z = zeros(1,13)
    itr = 1
    while itr < 5
        l = rand(leaf_r)
        t = rand(tree_r)
        d = rand(depth_r)
        r = rand(split_r)
        for s in rs
            println("itr=", itr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
            #MaxFeat = Int64(9)
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
            reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, n_jobs=-1, oob_score =true, random_state=s, class_weight=Dict(0=>0.529, 1=>9.097))
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
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                z[1,7] = avgScore(f1_10_train, 3)
                z[1,8] = score(reg, Matrix(Xx_test), Vector(Yy_test))
                z[1,9] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z[1,10] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z[1,11] = s
                z[1,12] = d
                z[1,13] = r
                println(z[end, :])
            else
                itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                traincvtrain = avgScore(f1_10_train, 3) 
                itest = score(reg, Matrix(Xx_test), Vector(Yy_test))
                f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z = vcat(z, [l t itrain jtrain ival jval traincvtrain itest f1s mccs s d r])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], f1_train = z[:,3], mcc_train = z[:,4], f1_val = z[:,5], mcc_val = z[:,6], f1_3Ftrain = z[:,7], acc_pest = z[:,8], f1_pest = z[:,9], mcc_pest = z[:,10], state = z[:,11], depth = z[:,12], minSampleSplit = z[:,13])
    z_df_sorted = sort(z_df, [:mcc_pest, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end =#

describe((inputDB))[10:12, :]
function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)
    #leaf_r = vcat(collect(2:2:20))
    leaf_r = vcat(collect(12:2:22))
    #leaf_r = vcat(collect(24:1:26))
    #tree_r = vcat(collect(50:50:400), collect(500:100:1000))
    tree_r = vcat(collect(350:50:550))
    #tree_r = vcat(collect(350:25:400), collect(500:50:1000))
    depth_r = vcat(collect(30:10:100))
    #depth_r = vcat(collect(50:5:70))
    split_r = vcat(collect(2:2:10))
    #split_r = vcat(collect(5:1:7))
    #rs = vcat(1, 42)
    rs = vcat(42)
    z = zeros(1,13)
    itr = 1
    while itr < 17
        l = rand(leaf_r)
        t = rand(tree_r)
        d = rand(depth_r)
        r = rand(split_r)
        for s in rs
            println("itr=", itr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
            println("## loading in data ##")
            M_train = inputDB
            M_val = inputDB_test
            M_pest = inputDB_pest
            #Xx_train = deepcopy(M_train[:, vcat(collect(5:12), end-3)])
            Xx_train = deepcopy(M_train[:, vcat(collect(5:10), 13, end-3)])
            Yy_train = deepcopy(M_train[:, end-2])
            #Xx_val = deepcopy(M_val[:, vcat(collect(5:12), end-3)])
            Xx_val = deepcopy(M_val[:, vcat(collect(5:10), 13, end-3)])
            Yy_val = deepcopy(M_val[:, end-2])
            #Xx_test = deepcopy(M_pest[:, vcat(collect(5:12), end-1)])
            Xx_test = deepcopy(M_pest[:, vcat(collect(5:10), 13, end-1)])
            Yy_test = deepcopy(M_pest[:, end])
            println("## Classification ##")
            reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, n_jobs=-1, oob_score =true, random_state=s, class_weight=Dict(0=>0.625, 1=>2.501))
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
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                z[1,7] = avgScore(f1_10_train, 3)
                z[1,8] = score(reg, Matrix(Xx_test), Vector(Yy_test))
                z[1,9] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z[1,10] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z[1,11] = s
                z[1,12] = d
                z[1,13] = r
                println(z[end, :])
            else
                itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                traincvtrain = avgScore(f1_10_train, 3) 
                itest = score(reg, Matrix(Xx_test), Vector(Yy_test))
                f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z = vcat(z, [l t itrain jtrain ival jval traincvtrain itest f1s mccs s d r])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], f1_train = z[:,3], mcc_train = z[:,4], f1_val = z[:,5], mcc_val = z[:,6], f1_3Ftrain = z[:,7], acc_pest = z[:,8], f1_pest = z[:,9], mcc_pest = z[:,10], state = z[:,11], depth = z[:,12], minSampleSplit = z[:,13])
    z_df_sorted = sort(z_df, [:mcc_pest, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatioFE2.csv"
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
      n_estimators = 600, 
      max_depth = 60, 
      min_samples_leaf = 26, 
      #max_features = Int64(9), 
      min_samples_split = 6, 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42, 
      class_weight= Dict(0=>0.625, 1=>2.501)
      )

fit!(model, Matrix(inputDB[:, vcat(collect(5:12), end-3)]), Vector(inputDB[:, end-2]))
# --------------------------------------------------------------------------------------------------
fit!(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-2]))
# --------------------------------------------------------------------------------------------------
fit!(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]), Vector(inputDBInputDB_test[:, end-2]))
# --------------------------------------------------------------------------------------------------
fit!(model, Matrix(inputDBInputDB_test[:, 5:12]), Vector(inputDBInputDB_test[:, end-2]))

# saving model
modelSavePath = "F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatio.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatio.joblib"
jl.dump(model, modelSavePath, compress = 5)

describe((inputDB_pest))[vcat(collect(5:12), end-1), :]

#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio.joblib")
size(model)
# training performace, withDeltaRi vs. withoutDeltaRi
predictedTPTN_train = predict(model,  Matrix(inputDB[:, vcat(collect(5:12), end-3)]))
inputDB[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio.joblib")
size(model)
predictedTPTN_train = predict(model, Matrix(inputDB[:, 5:12]))
inputDB[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatio.joblib")
size(model)
predictedTPTN_train = predict(model,  Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]))
inputDBInputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 4103848 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDBInputDB_test)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatio.joblib")
size(model)
predictedTPTN_train = predict(model, Matrix(inputDBInputDB_test[:, 5:12]))
inputDBInputDB_test[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 4103848 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDBInputDB_test)

# ==================================================================================================
inputDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
# 1.0, 0.05078740133496676, 0.2253606028900499; 1, 0.04918402791526732, 0.2217747233461634; 1, 0.04063321066389528, 0.20157681082876394; 1, 0.023676257463270747, 0.15387091168661718; 1, 0.14291923615582694, 0.37804660579858
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
# 0.06811669590416691; 0.09486528526360449; 0.2417866054361255; 0.5490194775471486; -0.9752121262865694
rSquare_train = rSquareDetermination(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
#= ## accuracy, 0.9977639885497694
acc1_train = score(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]), Vector(inputDB_withDeltaRiTPTN[:, end-3]))
# 0.938269856354399, 0.9336424942432107, 0.9370105577882192
acc5_train = cross_val_score(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]), Vector(inputDB_withDeltaRiTPTN[:, end-3]); cv = 3)
# 0.9363076361286096
avgAcc_train = avgAcc(acc5_train, 3) =#
# 3283078 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]))
# 0.6835861964267077; 0.6904598963694814; 0.730088782059064; 0.8227306831839195; 0.4346691438258374
f1_train = f1_score(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
# 0.7005990247521975; 0.7065755356627788; 0.74172739803641; 0.8253813934249877; 0.4853940601927635
mcc_train = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])

inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 3283078 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_withDeltaRiTPTN)

describe((inputDB_withDeltaRiTPTN))[end-5:end, :]
# --------------------------------------------------------------------------------------------------
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
# 1, 0.12646120500335356, 0.3556138425361892; 1, 0.12643957895608937, 0.35558343459178376; 1, 0.11452179936023452, 0.33841069628520093; 1, 0.09556854878257538, 0.3091416322376774; 1, 0.14312514049315916, 0.37831883444147896
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])
# -0.8622447329405676; -0.8620550173746595; -0.7608063359872013; -0.5651364546851403; -0.9769918561609672
rSquare_train = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])

# 3283078 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withoutDeltaRiTPTN[:, 5:12]))
# 0.464792327326166; 0.464841770887184; 0.4895904603655557; 0.5347700310045566; 0.4341600927711032
f1_train = f1_score(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])
# 0.5118613741680419; 0.511910767768887; 0.5335587435905159; 0.5726139077074558; 0.48480309322890747
mcc_train = matthews_corrcoef(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])

inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 3283078 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_withoutDeltaRiTPTN)

describe((inputDB_withoutDeltaRiTPTN))[end-5:end, :]

# --------------------------------------------------------------------------------------------------
inputWholeDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
# 1, 0.04544661498184143, 0.21318211693723615; 1, 0.0439792117056967, 0.20971221162749845; 1, 0.03760933640817106, 0.1939312672267447; 1, 0.022037609580081913, 0.14845069747253434; 1, 0.1429144061865839, 0.37804021768402357
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])
# 0.1582952037998726; 0.18347495500847466; 0.29514567107436496; 0.57966461677657; -0.9751673476664005
rSquare_train = rSquareDetermination(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])

# 4103848 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputWholeDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]))
# 0.707165960119328; 0.7138862291776314; 0.7450565823313804; 0.8329599332127864; 0.4346708795726432
f1_train = f1_score(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])
# 0.7212525691969476; 0.7271328609828559; 0.7549946622183068; 0.8348711000218666; 0.48536841409764003
mcc_train = matthews_corrcoef(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])

inputWholeDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputWholeDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 4103848 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputWholeDB_withDeltaRiTPTN)

describe((inputWholeDB_withDeltaRiTPTN))[end-5:end, :]

# --------------------------------------------------------------------------------------------------
inputWholeDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
# 1, 0.12467445188028406, 0.35309269587501246; 1, 0.12464837878985771, 0.35305577291676976; 1, 0.1134832479175642, 0.33687274736547657; 1, 0.09507710811901415, 0.3083457606632758; 1, 0.14309837986202217, 0.37828346495983955
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])
# -0.8480738936391097; -0.8478486881195764; -0.7510920417232618; -0.5593579616994719; -0.9766652449219255
rSquare_train = rSquareDetermination(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])

# 4103848 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputWholeDB_withoutDeltaRiTPTN[:, 5:12]))
# 0.46833847525487066; 0.46839607837431696; 0.49189152625216576; 0.5360819348962854; 0.43424252116578477
f1_train = f1_score(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])
# 0.5149516079309533; 0.5150068737461049; 0.5355584377597136; 0.5737481606246932; 0.48488752192250767
mcc_train = matthews_corrcoef(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])

inputWholeDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputWholeDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 4103848 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputWholeDB_withoutDeltaRiTPTN)

describe((inputWholeDB_withoutDeltaRiTPTN))[end-5:end, :]


# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatio.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatio.joblib")
size(modelRF_TPTN)

# ==================================================================================================

describe((inputDB_pest))[end-5:end, :]

predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_pest[:, vcat(collect(5:12), end-1)]))
inputDB_pest[!, "withDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 136678 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_pest)
# --------------------------------------------------------------------------------------------------
predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_pest[:, 5:12]))
inputDB_pest[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 136678 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_pest)

# ==================================================================================================
inputPestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
describe((inputPestDB_withDeltaRiTPTN))[end-5:end, :]

# 1, 0.05357116726905574, 0.2314544604648088; 1, 0.053015115819663734, 0.2302501157864285; 1, 0.05290536882307321, 0.23001167105839043; 1, 0.05167620246125931, 0.22732400326683347
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])
# 0.283526929864932; 0.29096382544432287; 0.29474374285526506; 0.327994115543199
rSquare_val = rSquareDetermination(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])

# 136678 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-2)]))
# 0.6718946047678795; 0.6733387431250564; 0.6404097667710975; 0.7526527753458239
f1_test = f1_score(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])
# 0.6427315488312465; 0.6444940977121384; 0.6162816101472732; 0.7481962160220814
mcc_test = matthews_corrcoef(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])

inputPestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 136678 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputPestDB_withDeltaRiTPTN)

describe((inputPestDB_withDeltaRiTPTN))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
inputPestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
describe((inputPestDB_withoutDeltaRiTPTN))[end-5:end, :]

# 1, 0.05104698634747362, 0.2259358013849811; 1, 0.05073237829058078, 0.22523849202696414; 1, 0.05185911412224352, 0.2277259627759723; 1, 0.05176400005853173, 0.22751703245808155
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])
# 0.33156593391445466; 0.33556877227653015; 0.31299637888432974; 0.3268760235509488
rSquare_val = rSquareDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])

# 136678 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withoutDeltaRiTPTN[:, 5:12]))
# 0.748240897773608; 0.7495846876128566; 0.7263954296302014; 0.7522672362477678
f1_test = f1_score(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])
# 0.7388999236740152; 0.7403152751341858; 0.706988498838258; 0.7477759798945115
mcc_test = matthews_corrcoef(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])

inputPestDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 136678 x 19+2 df 
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
CSV.write(savePath, inputPestDB_withoutDeltaRiTPTN)

describe((inputPestDB_withoutDeltaRiTPTN))[end-4:end, :]
