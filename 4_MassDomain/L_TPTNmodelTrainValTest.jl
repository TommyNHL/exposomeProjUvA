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
# inputing 820536 x 4+8+1+2+1+1+2+2 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestYesDFwithhl.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB_test, 1)
    inputDB_test[i, "MS1Error"] = abs(inputDB_test[i, "MS1Error"])
    inputDB_test[i, "DeltaRi"] = abs(inputDB_test[i, "DeltaRi"])
end
# save, ouputing 409002 x 21 df, 0:372896; 1:36106 = 0.5484; 5.6639
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDFwithhl0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_test)
inputDB_test[inputDB_test.LABEL .== 1, :]

# inputing 3282229 x 4+8+1+2+1+1+2 df
inputDB = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainYesDFwithhl.csv", DataFrame)
sort!(inputDB, [:ENTRY])
inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB, 1)
    inputDB[i, "MS1Error"] = abs(inputDB[i, "MS1Error"])
    inputDB[i, "DeltaRi"] = abs(inputDB[i, "DeltaRi"])
end
# save, ouputing 1636788 x 21 df, 0:1492439; 1:144349 = 0.5484; 5.6696
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB)
inputDB[inputDB.LABEL .== 1, :]

# 2045790 x 21 df; 
# 409002+1636788= 2045790, 0:1865335; 1:180455 = 0.5484; 5.6684
inputDBInputDB_test = vcat(inputDB, inputDB_test)
sort!(inputDBInputDB_test, [:ENTRY])
inputDBInputDB_test[inputDBInputDB_test.LABEL .== 1, :]

# 136678 x 19 df
inputDB_pest = CSV.read("F:\\UvA\\dataframeTPTNModeling_pest.csv", DataFrame)
sort!(inputDB_pest, [:ENTRY])
inputDB_pest = inputDB_pest[inputDB_pest.FinalScoreRatio .>= float(0.5), :]
inputDB_pest = inputDB_pest[inputDB_pest.Leverage .<= 0.14604417882015916, :]
for i = 1:size(inputDB_pest, 1)
    inputDB_pest[i, "MS1Error"] = abs(inputDB_pest[i, "MS1Error"])
    inputDB_pest[i, "DeltaRi"] = abs(inputDB_pest[i, "DeltaRi"])
end
# save, ouputing 61988 x 19 df, 0:53142; 1:8846 = 
savePath = "F:\\UvA\\dataframeTPTNModeling_pestDFwithhl0d5FinalScoreRatio.csv"
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
describe((inputDB_test))[vcat(collect(5:12), end-5), :]

function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)
    #leaf_r = vcat(collect(2:2:16))
    leaf_r = vcat(collect(5:1:7))
    #tree_r = vcat(collect(50:50:400))
    tree_r = vcat(collect(325:25:425))
    #depth_r = vcat(collect(30:10:100))
    depth_r = vcat(collect(40:5:80))
    #split_r = vcat(collect(2:1:10))
    split_r = vcat(collect(6:1:12))
    model_r = vcat(9, 8)
    rs = 42
    z = zeros(1,14)
    itr = 1
    while itr < 33
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
                Xx_train = deepcopy(M_train[:, vcat(collect(5:12))])
                Xx_val = deepcopy(M_val[:, vcat(collect(5:12))])
                Xx_test = deepcopy(M_pest[:, vcat(collect(5:12))])
            elseif mod == 9
                Xx_train = deepcopy(M_train[:, vcat(collect(5:12), end-5)])
                Xx_val = deepcopy(M_val[:, vcat(collect(5:12), end-5)])
                Xx_test = deepcopy(M_pest[:, vcat(collect(5:12), end-2)])
            end
            Yy_train = deepcopy(M_train[:, end-4])
            Yy_val = deepcopy(M_val[:, end-4])
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

#= optiSearch_df = optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatio_RFwithhlnewCompare2.csv"
CSV.write(savePath, optiSearch_df) =#

function optimLR(inputDB, inputDB_test, inputDB_pest)
    penalty_r = vcat("l1", "l2")
    solver_rs = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    solver_r = vcat(collect(1:1:5))
    c_values_r = vcat(1000, 100, 10, 1.0)
    model_r = vcat(9, 8)
    rs = 42
    z = zeros(1,13)
    itr = 1
    pn = float(0)
    while itr < 17
        p = rand(penalty_r)
        s = rand(solver_r)
        c = rand(c_values_r)
        if solver_rs[s] == "lbfgs" || solver_rs[s] == "newton-cg" || solver_rs[s] == "sag"
            p = "l2"
            pn = float(2)
        end
        if p == "l1"
            pn = float(1)
        elseif p == "l2"
            pn = float(2)
        end
        for mod in model_r
            println("itr=", itr, ", penalty=", p, ", solver=", s, ", C=", c)
            println("## loading in data ##")
            M_train = inputDB
            M_val = inputDB_test
            M_pest = inputDB_pest
            if mod == 8
                Xx_train = deepcopy(M_train[:, vcat(collect(5:12))])
                Xx_val = deepcopy(M_val[:, vcat(collect(5:12))])
                Xx_test = deepcopy(M_pest[:, vcat(collect(5:12))])
            elseif mod == 9
                Xx_train = deepcopy(M_train[:, vcat(collect(5:12), end-5)])
                Xx_val = deepcopy(M_val[:, vcat(collect(5:12), end-5)])
                Xx_test = deepcopy(M_pest[:, vcat(collect(5:12), end-2)])
            end
            Yy_train = deepcopy(M_train[:, end-4])
            Yy_val = deepcopy(M_val[:, end-4])
            Yy_test = deepcopy(M_pest[:, end-1])
            println("## Classification ##")
            reg = LogisticRegression(penalty=p, C=c, solver=solver_rs[s], max_iter=5000, random_state=rs, class_weight=Dict(0=>0.5484, 1=>5.6684))
            println("## fit ##")
            fit!(reg, Matrix(Xx_train), Vector(Yy_train))
            if itr == 1
                z[1,1] = pn
                z[1,2] = c
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
                z[1,12] = s
                z[1,13] = mod
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
                z = vcat(z, [pn c itrain jtrain ival jval traincvtrain itest f1s mccs rs s mod])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(Penalty = z[:,1], C_value = z[:,2], f1_train = z[:,3], mcc_train = z[:,4], f1_val = z[:,5], mcc_val = z[:,6], f1_3Ftrain = z[:,7], acc_pest = z[:,8], f1_pest = z[:,9], mcc_pest = z[:,10], state = z[:,11], Solver = z[:,12], model = z[:,13])
    z_df_sorted = sort(z_df, [:f1_3Ftrain, :f1_pest, ], rev=true)
    return z_df_sorted
end

#= optiSearch_df = optimLR(inputDB, inputDB_test, inputDB_pest)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatio_LRwithhlnewCompare1.csv"
CSV.write(savePath, optiSearch_df) =#

Yy_train = deepcopy(inputDB[:, end-4])
sampleW = []
for w in Vector(Yy_train)
    if w == 0
        push!(sampleW, 0.5484)
    elseif w == 1
        push!(sampleW, 5.6684)
    end
end

Yy_trainWhole = deepcopy(inputDBInputDB_test[:, end-4])
sampleTW = []
for w in Vector(Yy_trainWhole)
    if w == 0
        push!(sampleTW, 0.5484)
    elseif w == 1
        push!(sampleTW, 5.6684)
    end
end

function optimGradientBoostClass(inputDB, inputDB_test, inputDB_pest)
    lr_r = vcat(collect(0.1:0.1:0.5))
    leaf_r = vcat(collect(16:2:32))
    tree_r = vcat(8, 16, 24, 32, 64)
    depth_r = vcat(collect(10:2:20))
    split_r = vcat(collect(12:2:20))
    model_r = vcat(9, 8)
    rs = 42
    z = zeros(1,15)
    itr = 1
    while itr < 65
        lr = rand(lr_r)
        l = rand(leaf_r)
        t = rand(tree_r)
        d = rand(depth_r)
        r = rand(split_r)
        for mod in model_r
            println("itr=", itr, ", lr=", lr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
            println("## loading in data ##")
            M_train = inputDB
            M_val = inputDB_test
            M_pest = inputDB_pest
            if mod == 8
                Xx_train = deepcopy(M_train[:, vcat(collect(5:12))])
                Xx_val = deepcopy(M_val[:, vcat(collect(5:12))])
                Xx_test = deepcopy(M_pest[:, vcat(collect(5:12))])
            elseif mod == 9
                Xx_train = deepcopy(M_train[:, vcat(collect(5:12), end-5)])
                Xx_val = deepcopy(M_val[:, vcat(collect(5:12), end-5)])
                Xx_test = deepcopy(M_pest[:, vcat(collect(5:12), end-2)])
            end
            Yy_train = deepcopy(M_train[:, end-4])
            Yy_val = deepcopy(M_val[:, end-4])
            Yy_test = deepcopy(M_pest[:, end-1])
            println("## Classification ##")
            reg = GradientBoostingClassifier(learning_rate=lr, n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, random_state=rs)
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
                z[1,12] = rs
                z[1,13] = d
                z[1,14] = r
                z[1,15] = mod
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
                z = vcat(z, [lr l t itrain jtrain ival jval traincvtrain itest f1s mccs rs d r mod])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(learnRate = z[:,1], leaves = z[:,2], trees = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_val = z[:,6], mcc_val = z[:,7], f1_3Ftrain = z[:,8], acc_pest = z[:,9], f1_pest = z[:,10], mcc_pest = z[:,11], state = z[:,12], depth = z[:,13], minSampleSplit = z[:,14], model = z[:,15])
    z_df_sorted = sort(z_df, [:f1_3Ftrain, :f1_pest], rev=true)
    return z_df_sorted
end

#= optiSearch_df = optimGradientBoostClass(inputDB, inputDB_test, inputDB_pest)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatio_GBMswithhlnewCompare1.csv"
CSV.write(savePath, optiSearch_df) =#


model = RandomForestClassifier(
      n_estimators = 350, 
      max_depth = 60, 
      min_samples_leaf = 7, 
      min_samples_split = 11, 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42, 
      class_weight= Dict(0=>0.5484, 1=>5.6684)
      )


fit!(model, Matrix(inputDB[:, vcat(collect(5:12), end-5)]), Vector(inputDB[:, end-4]))
# --------------------------------------------------------------------------------------------------
fit!(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-4]))
# --------------------------------------------------------------------------------------------------

# saving model
modelSavePath = "F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio_RFwithhlnew.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
modelSavePath = "F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio_RFwithhlnew.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------

describe((inputDB))[vcat(collect(5:12), end-5, end-4), :]

#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio_RFwithhlnew.joblib")
size(model)
# training performace, withDeltaRi vs. withoutDeltaRi
predictedTPTN_train = predict(model, Matrix(inputDB[:, vcat(collect(5:12), end-5)]))
inputDB[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 1636788 x 22 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
model = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio_RFwithhlnew.joblib")
size(model)
predictedTPTN_train = predict(model, Matrix(inputDB[:, 5:12]))
inputDB[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 1636788 x 22 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB)

# ==================================================================================================
inputDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv", DataFrame)
# RF: 1, 0.06468400305965097, 0.2543304996646115
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])
# RF: 0.2353154894389773
rSquare_train = rSquareDetermination(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])

# 1636788 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-6)]))
# RF: 0.7316071528161553
f1_train = f1_score(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])
# RF: 0.7319669255605697
mcc_train = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])

inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 1636788 x (22+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_withDeltaRiTPTN)

describe((inputDB_withDeltaRiTPTN))[end-5:end, :]
# --------------------------------------------------------------------------------------------------
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv", DataFrame)
# RF: 1, 0.12487811494219166, 0.3533809770519512
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])
# RF: -0.3010813175216762
rSquare_train = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])

# 1636788 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withoutDeltaRiTPTN[:, 5:12]))
# RF: 0.5853294354650561
f1_train = f1_score(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])
# RF: 0.5974037125510779
mcc_train = matthews_corrcoef(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])

inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 1636788 x (22+2)
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_withoutDeltaRiTPTN)

describe((inputDB_withoutDeltaRiTPTN))[end-5:end, :]

# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio_RFwithhlnew.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio_RFwithhlnew.joblib")
size(modelRF_TPTN)

# ==================================================================================================

describe((inputDB_test))[end-5:end, :]

predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-5)]))
inputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 409002 x 22 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_test)
# --------------------------------------------------------------------------------------------------
predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]))
inputDB_test[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 409002 x 22 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_test)
# ==================================================================================================
inputTestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv", DataFrame)
describe((inputTestDB_withDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.09148855017823873, 0.3024707426814017
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])
# RF: -0.0561496284297649
rSquare_val = rSquareDetermination(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])

# 409002 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputTestDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-6)]))
# RF: 0.641095732742497
f1_test = f1_score(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])
# RF: 0.6337610017111366
mcc_test = matthews_corrcoef(inputTestDB_withDeltaRiTPTN[:, end-5], inputTestDB_withDeltaRiTPTN[:, end])

inputTestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputTestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 409002 x 22+2 df
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputTestDB_withDeltaRiTPTN)

describe((inputTestDB_withDeltaRiTPTN))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
inputTestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv", DataFrame)
describe((inputTestDB_withoutDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.1461777693018616, 0.3823320144872276
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputTestDB_withoutDeltaRiTPTN[:, end-5], inputTestDB_withoutDeltaRiTPTN[:, end])
# RF: -0.4824047449575817
rSquare_val = rSquareDetermination(inputTestDB_withoutDeltaRiTPTN[:, end-5], inputTestDB_withoutDeltaRiTPTN[:, end])

# 409002 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputTestDB_withoutDeltaRiTPTN[:, 5:12]))
# RF: 0.5302460066157001
f1_test = f1_score(inputTestDB_withoutDeltaRiTPTN[:, end-5], inputTestDB_withoutDeltaRiTPTN[:, end])
# RF: 0.53207109908907
mcc_test = matthews_corrcoef(inputTestDB_withoutDeltaRiTPTN[:, end-5], inputTestDB_withoutDeltaRiTPTN[:, end])

inputTestDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputTestDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 409002 x 22+2 df 
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputTestDB_withoutDeltaRiTPTN)

describe((inputTestDB_withoutDeltaRiTPTN))[end-4:end, :]

# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio_RFwithhlnew.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio_RFwithhlnew.joblib")
size(modelRF_TPTN)

# ==================================================================================================

describe((inputDB_pest))[end-2:end, :]

predictedTPTN_pest = predict(modelRF_TPTN, Matrix(inputDB_pest[:, vcat(collect(5:12), end-2)]))
inputDB_pest[!, "withDeltaRipredictTPTN"] = predictedTPTN_pest
# save, ouputing testSet df 61988 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_pest)
# --------------------------------------------------------------------------------------------------
predictedTPTN_pest = predict(modelRF_TPTN, Matrix(inputDB_pest[:, 5:12]))
inputDB_pest[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_pest
# save, ouputing testSet df 61988 x 19 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_pest)
# ==================================================================================================
inputPestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv", DataFrame)
describe((inputPestDB_withDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.0640285216493515, 0.2530385773935498
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])
# RF: 0.476637814683172
rSquare_val = rSquareDetermination(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])

# 61988 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-3)]))
# RF: 0.7761800033835223
f1_test = f1_score(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])
# RF: 0.7388251700763637
mcc_test = matthews_corrcoef(inputPestDB_withDeltaRiTPTN[:, end-2], inputPestDB_withDeltaRiTPTN[:, end])

inputPestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 61988 x 19+2 df
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputPestDB_withDeltaRiTPTN)

describe((inputPestDB_withDeltaRiTPTN))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
inputPestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_new0d5FinalScoreRatio.csv", DataFrame)
describe((inputPestDB_withoutDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.0633832354649287, 0.2517602738021404
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-2], inputPestDB_withoutDeltaRiTPTN[:, end])
# RF: 0.48506496682441214
rSquare_val = rSquareDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-2], inputPestDB_withoutDeltaRiTPTN[:, end])

# 61988 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withoutDeltaRiTPTN[:, 5:12]))
# RF: 0.7973593274537107
f1_test = f1_score(inputPestDB_withoutDeltaRiTPTN[:, end-2], inputPestDB_withoutDeltaRiTPTN[:, end])
# RF: 0.764245729777751
mcc_test = matthews_corrcoef(inputPestDB_withoutDeltaRiTPTN[:, end-2], inputPestDB_withoutDeltaRiTPTN[:, end])

inputPestDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 61988 x 19+2 df 
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_new0d5FinalScoreRatio.csv"
CSV.write(savePath, inputPestDB_withoutDeltaRiTPTN)

describe((inputPestDB_withoutDeltaRiTPTN))[end-4:end, :]
