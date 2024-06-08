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

describe((inputDB_test))[12:14, :]
# inputing 820770 x 4+8+1+2+1+1+2 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestYesDFwithhl.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
insertcols!(inputDB_test, 10, ("MatchRatio"=>float(0)))
inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB_test, 1)
    inputDB_test[i, "RefMatchFragRatio"] = log10(inputDB_test[i, "RefMatchFragRatio"])
    inputDB_test[i, "UsrMatchFragRatio"] = log10(inputDB_test[i, "UsrMatchFragRatio"])
    inputDB_test[i, "FinalScoreRatio"] = log10(inputDB_test[i, "FinalScoreRatio"])
    inputDB_test[i, "MatchRatio"] = inputDB_test[i, "DirectMatch"] - inputDB_test[i, "ReversMatch"]
end
# save, ouputing 409139 x 22 df, 0:373024; 1:36115 = 0.5484; 5.6644
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDFwithhl0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB_test)
inputDB_test[inputDB_test.LABEL .== 1, :]

# inputing 3283078 x 4+8+1+2+1+1+2+2 df
inputDB = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainYesDFwithhl.csv", DataFrame)
sort!(inputDB, [:ENTRY])
insertcols!(inputDB, 10, ("MatchRatio"=>float(0)))
inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB, 1)
    inputDB[i, "RefMatchFragRatio"] = log10(inputDB[i, "RefMatchFragRatio"])
    inputDB[i, "UsrMatchFragRatio"] = log10(inputDB[i, "UsrMatchFragRatio"])
    inputDB[i, "FinalScoreRatio"] = log10(inputDB[i, "FinalScoreRatio"])
    inputDB[i, "MatchRatio"] = inputDB[i, "DirectMatch"] - inputDB[i, "ReversMatch"]
end
# save, ouputing 1637225 x 22 df, 0:1492833; 1:144392 = 0.5484; 5.6656
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB)
inputDB[inputDB.LABEL .== 1, :]

# 2046364 x 21 df; 
# 409139+1637225= 2046364, 0:1865857; 1:180507 = 0.5484; 5.6684
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
end
# save, ouputing 62008 x 17 df, 0:53162; 1:8846 = 
savePath = "F:\\UvA\\dataframeTPTNModeling_pestDFwithhl0d5FinalScoreRatioDE.csv"
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
describe((inputDB_pest))[vcat(collect(5:10), 13, end-2), :]

function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)
    leaf_r = vcat(collect(2:2:16))
    tree_r = vcat(collect(50:50:400))
    depth_r = vcat(collect(30:10:100))
    split_r = vcat(collect(2:1:10))
    model_r = vcat(9, 8)
    rs = 42
    z = zeros(1,14)
    itr = 1
    while itr < 65
        l = rand(leaf_r)
        t = rand(tree_r)
        d = rand(depth_r)
        r = rand(split_r)
        for mod in model_r
            println("itr=", itr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
            println("## loading in data ##")
            M_train = inputDB
            M_val = inputDB_test
            M_pest = inputDB_pest
            if mod == 8
                Xx_train = deepcopy(M_train[:, vcat(collect(5:10), 13)])
                Xx_val = deepcopy(M_val[:, vcat(collect(5:10), 13)])
                Xx_test = deepcopy(M_pest[:, vcat(collect(5:10), 13)])
            elseif mod == 9
                Xx_train = deepcopy(M_train[:, vcat(collect(5:10), 13, end-3)])
                Xx_val = deepcopy(M_val[:, vcat(collect(5:10), 13, end-3)])
                Xx_test = deepcopy(M_pest[:, vcat(collect(5:10), 13, end-2)])
            end
            Yy_train = deepcopy(M_train[:, end-2])
            Yy_val = deepcopy(M_val[:, end-2])
            Yy_test = deepcopy(M_pest[:, end-1])
            println("## Classification ##")
            reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, n_jobs=-1, oob_score =true, random_state=rs, class_weight=Dict(0=>0.5484, 1=>5.6684))
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
                z[1,11] = rs
                z[1,12] = d
                z[1,13] = r
                z[1,14] = mod
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
                z = vcat(z, [l t itrain jtrain ival jval traincvtrain itest f1s mccs rs d r mod])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], f1_train = z[:,3], mcc_train = z[:,4], f1_val = z[:,5], mcc_val = z[:,6], f1_3Ftrain = z[:,7], acc_pest = z[:,8], f1_pest = z[:,9], mcc_pest = z[:,10], state = z[:,11], depth = z[:,12], minSampleSplit = z[:,13], model = z[:,14])
    z_df_sorted = sort(z_df, [:f1_3Ftrain, :f1_pest], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatioDE_RFwithhlnewCompare1.csv"
CSV.write(savePath, optiSearch_df)

Yy_train = deepcopy(inputDB[:, end-2])
sampleW = []
for w in Vector(Yy_train)
    if w == 0
        push!(sampleW, 0.5484)
    elseif w == 1
        push!(sampleW, 5.6684)
    end
end 

Yy_trainWhole = deepcopy(inputDBInputDB_test[:, end-2])
sampleTW = []
for w in Vector(Yy_trainWhole)
    if w == 0
        push!(sampleTW, 0.5484)
    elseif w == 1
        push!(sampleTW, 5.6684)
    end
end 

function optimGradientBoostClass(inputDB, inputDB_test, inputDB_pest)
    #lr_r = vcat(1, 0.5, 0.25, 0.1, 0.05)
    lr_r = vcat(0.5, 0.25, 0.1, 0.05)
    #leaf_r = vcat(collect(2:2:24))
    leaf_r = vcat(collect(12:2:32))
    #tree_r = vcat(4, 8, 16, 32, 64, 100)
    tree_r = vcat(4, 8, 16, 32, 64, 128)
    #depth_r = vcat(collect(2:2:24))
    depth_r = vcat(collect(2:2:18))
    #split_r = vcat(collect(2:2:12))
    split_r = vcat(collect(2:2:18))
    #rs = vcat(1, 42)
    rs = vcat(42)
    z = zeros(1,14)
    itr = 1
    while itr < 9
        lr = rand(lr_r)
        l = rand(leaf_r)
        t = rand(tree_r)
        d = rand(depth_r)
        r = rand(split_r)
        for s in rs
            println("itr=", itr, ", lr=", lr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
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
            Xx_test = deepcopy(M_pest[:, vcat(collect(5:10), 13, end-3)])
            Yy_test = deepcopy(M_pest[:, end])
            println("## Classification ##")
            reg = GradientBoostingClassifier(learning_rate=lr, n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, random_state=s)
            println("## fit ##")
            fit!(reg, Matrix(Xx_train), Vector(Yy_train), sample_weight=sampleW)
            if itr == 1
                z[1,1] = lr
                z[1,2] = l
                z[1,3] = t
                z[1,4] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                z[1,5] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)))
                z[1,6] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                z[1,7] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                z[1,8] = avgScore(f1_10_train, 3)
                z[1,9] = score(reg, Matrix(Xx_test), Vector(Yy_test))
                z[1,10] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z[1,11] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)))
                z[1,12] = s
                z[1,13] = d
                z[1,14] = r
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
                z = vcat(z, [lr l t itrain jtrain ival jval traincvtrain itest f1s mccs s d r])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(learnRate = z[:,1], leaves = z[:,2], trees = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_val = z[:,6], mcc_val = z[:,7], f1_3Ftrain = z[:,8], acc_pest = z[:,9], f1_pest = z[:,10], mcc_pest = z[:,11], state = z[:,12], depth = z[:,13], minSampleSplit = z[:,14])
    z_df_sorted = sort(z_df, [:mcc_pest, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimGradientBoostClass(inputDB, inputDB_test, inputDB_pest)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatioDE2_GBMs.csv"
CSV.write(savePath, optiSearch_df)

model = RandomForestClassifier(
      n_estimators = 425, 
      max_depth = 100, 
      min_samples_leaf = 28, 
      #max_features = Int64(9), 
      min_samples_split = 12, 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42, 
      class_weight= Dict(0=>0.625, 1=>2.501)
      )

model = GradientBoostingClassifier(
      learning_rate = 0.25, 
      n_estimators = 8, 
      max_depth = 18, 
      min_samples_leaf = 18, 
      min_samples_split = 18, 
      random_state = 42
      )

#fit!(model, Matrix(inputDB[:, vcat(collect(5:12), end-3)]), Vector(inputDB[:, end-2]))
fit!(model, Matrix(inputDB[:, vcat(collect(5:10), 13, end-3)]), Vector(inputDB[:, end-2]), sample_weight=sampleW)
# --------------------------------------------------------------------------------------------------
#fit!(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-2]))
fit!(model, Matrix(inputDB[:, vcat(collect(5:10), 13)]), Vector(inputDB[:, end-2]), sample_weight=sampleW)
# --------------------------------------------------------------------------------------------------
#fit!(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]), Vector(inputDBInputDB_test[:, end-2]))
fit!(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:10), 13, end-3)]), Vector(inputDBInputDB_test[:, end-2]), sample_weight=sampleTW)
# --------------------------------------------------------------------------------------------------
#fit!(model, Matrix(inputDBInputDB_test[:, 5:12]), Vector(inputDBInputDB_test[:, end-2]))
fit!(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:10), 13)]), Vector(inputDBInputDB_test[:, end-2]), sample_weight=sampleTW)

# saving model
modelSavePath = "F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib"
jl.dump(model, modelSavePath, compress = 5)

describe((inputDB_pest))[vcat(collect(5:12), end-1), :]

#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
size(model)
# training performace, withDeltaRi vs. withoutDeltaRi
#predictedTPTN_train = predict(model, Matrix(inputDB[:, vcat(collect(5:12), end-3)]))
predictedTPTN_train = predict(model, Matrix(inputDB[:, vcat(collect(5:10), 13, end-3)]))
inputDB[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
size(model)
#predictedTPTN_train = predict(model, Matrix(inputDB[:, 5:12]))
predictedTPTN_train = predict(model, Matrix(inputDB[:, vcat(collect(5:10), 13)]))
inputDB[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3283078 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
size(model)
#predictedTPTN_train = predict(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]))
predictedTPTN_train = predict(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:10), 13, end-3)]))
inputDBInputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 4103848 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputDBInputDB_test)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
size(model)
#predictedTPTN_train = predict(model, Matrix(inputDBInputDB_test[:, 5:12]))
predictedTPTN_train = predict(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:10), 13)]))
inputDBInputDB_test[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 4103848 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputDBInputDB_test)

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

# --------------------------------------------------------------------------------------------------
inputWholeDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv", DataFrame)
# 1, 0.09807394969809867, 0.31316760639967006
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])
# -0.1104071443342105
rSquare_train = rSquareDetermination(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])

# 4103848 × 2 Matrix
#pTP_train = predict_proba(model, Matrix(inputWholeDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]))
pTP_train = predict_proba(model, Matrix(inputWholeDB_withDeltaRiTPTN[:, vcat(collect(5:10), 13, end-4)]))
# 0.6302849658368903
f1_train = f1_score(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])
# 0.6279550932157474
mcc_train = matthews_corrcoef(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])

inputWholeDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputWholeDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 4103848 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputWholeDB_withDeltaRiTPTN)

describe((inputWholeDB_withDeltaRiTPTN))[end-5:end, :]

# --------------------------------------------------------------------------------------------------
inputWholeDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatioDE_GBMs.csv", DataFrame)
# 1, 0.11947630040403369, 0.345653439739913
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])
# -0.30096520557440654
rSquare_train = rSquareDetermination(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])

# 4103848 × 2 Matrix
#pTP_train = predict_proba(model, Matrix(inputWholeDB_withoutDeltaRiTPTN[:, 5:12]))
pTP_train = predict_proba(model, Matrix(inputWholeDB_withoutDeltaRiTPTN[:, vcat(collect(5:10), 13)]))
# 0.5781638744629825
f1_train = f1_score(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])
# 0.5755637742866574
mcc_train = matthews_corrcoef(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])

inputWholeDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputWholeDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 4103848 x (19+1+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatioDE_GBMs.csv"
CSV.write(savePath, inputWholeDB_withoutDeltaRiTPTN)

describe((inputWholeDB_withoutDeltaRiTPTN))[end-5:end, :]


# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatioDE_GBMs.joblib")
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
