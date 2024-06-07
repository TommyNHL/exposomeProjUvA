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
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDFwithhl.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB_test, 1)
    inputDB_test[i, "MS1Error"] = abs(inputDB_test[i, "MS1Error"])
    inputDB_test[i, "DeltaRi"] = abs(inputDB_test[i, "DeltaRi"])
end
# save, ouputing 409139 x 21 df, 0:373024; 1:36115 = 0.5484; 5.6644
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDFwithhl0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB_test)
inputDB_test[inputDB_test.LABEL .== 0, :]

# inputing 3283078 x 4+8+1+2+1+1+2 df
inputDB = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl.csv", DataFrame)
sort!(inputDB, [:ENTRY])
inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB, 1)
    inputDB[i, "MS1Error"] = abs(inputDB[i, "MS1Error"])
    inputDB[i, "DeltaRi"] = abs(inputDB[i, "DeltaRi"])
end
# save, ouputing 1637238 x 19 df, 0:1492833; 1:144392 = 0.5484; 5.6694
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl0d5FinalScoreRatio.csv"
CSV.write(savePath, inputDB)
inputDB[inputDB.LABEL .== 0, :]

# 4103848 x 19 df; 
# 409126+1637238= 2046364, 0:1865857; 1:180507 = 0.5484; 5.6684
inputDBInputDB_test = vcat(inputDB, inputDB_test)
sort!(inputDBInputDB_test, [:ENTRY])
inputDBInputDB_test[inputDBInputDB_test.LABEL .== 1, :]

# 136678 x 17 df
inputDB_pest = CSV.read("F:\\UvA\\dataframeTPTNModeling_pest.csv", DataFrame)
sort!(inputDB_pest, [:ENTRY])

inputDB_pest = inputDB_pest[inputDB_pest.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB_pest, 1)
    inputDB_pest[i, "MS1Error"] = abs(inputDB_pest[i, "MS1Error"])
    inputDB_pest[i, "DeltaRi"] = abs(inputDB_pest[i, "DeltaRi"])
end

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
describe((inputDB))[10:12, :]
#= function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest)
    #leaf_r = vcat(collect(2:2:20))
    #leaf_r = vcat(collect(12:2:22))
    leaf_r = vcat(collect(25:1:28))
    #tree_r = vcat(collect(50:50:400), collect(500:100:1000))
    tree_r = vcat(collect(400:25:500))
    #tree_r = vcat(collect(350:25:400), collect(500:50:1000))
    depth_r = vcat(collect(90:5:110))
    #depth_r = vcat(collect(50:5:70))
    #split_r = vcat(collect(2:2:14))
    split_r = vcat(collect(2:2:14))
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
            Xx_train = deepcopy(M_train[:, vcat(collect(5:12), end-3)])
            Yy_train = deepcopy(M_train[:, end-2])
            Xx_val = deepcopy(M_val[:, vcat(collect(5:12), end-3)])
            Yy_val = deepcopy(M_val[:, end-2])
            Xx_test = deepcopy(M_pest[:, vcat(collect(5:12), end-1)])
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
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatio5.csv"
CSV.write(savePath, optiSearch_df) =#

#= function optimLR(inputDB, inputDB_test, inputDB_pest)
    penalty_r = vcat("l1", "l2")
    solver_rs = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    solver_r = vcat(collect(1:1:5))
    c_values_r = vcat(1000, 100, 10, 1.0, 0.1, 0.01, 0.001)
    #rs = vcat(1, 42)
    rs = vcat(42)
    z = zeros(1,12)
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
        for r in rs
            println("itr=", itr, ", penalty=", p, ", solver=", s, ", C=", c)
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
            reg = LogisticRegression(penalty=p, C=c, solver=solver_rs[s], n_jobs=-1, max_iter=1000, random_state=r, class_weight=Dict(0=>0.625, 1=>2.501))
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
                z[1,11] = r
                z[1,12] = s
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
                z = vcat(z, [pn c itrain jtrain ival jval traincvtrain itest f1s mccs r s])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(Penalty = z[:,1], C_value = z[:,2], f1_train = z[:,3], mcc_train = z[:,4], f1_val = z[:,5], mcc_val = z[:,6], f1_3Ftrain = z[:,7], acc_pest = z[:,8], f1_pest = z[:,9], mcc_pest = z[:,10], state = z[:,11], Solver = z[:,12])
    z_df_sorted = sort(z_df, [:mcc_pest, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimLR(inputDB, inputDB_test, inputDB_pest)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatio_LR.csv"
CSV.write(savePath, optiSearch_df) =#

Yy_train = deepcopy(inputDB[:, end-2])
sampleW = []
for w in Vector(Yy_train)
    if w == 0
        push!(sampleW, 0.625)
    elseif w == 1
        push!(sampleW, 2.501)
    end
end 

Yy_trainWhole = deepcopy(inputDBInputDB_test[:, end-2])
sampleTW = []
for w in Vector(Yy_trainWhole)
    if w == 0
        push!(sampleTW, 0.625)
    elseif w == 1
        push!(sampleTW, 2.501)
    end
end 

function optimGradientBoostClass(inputDB, inputDB_test, inputDB_pest)
    #lr_r = vcat(1, 0.5, 0.25, 0.1, 0.05)
    #lr_r = vcat(0.5, 0.25, 0.1, 0.05)
    lr_r = vcat(collect(0.1:0.05:0.5))
    #leaf_r = vcat(collect(2:2:24))
    leaf_r = vcat(collect(10:1:35))
    #tree_r = vcat(4, 8, 16, 32, 64, 100)
    tree_r = vcat(16, 32)
    #depth_r = vcat(collect(2:2:24))
    depth_r = vcat(collect(16:1:18))
    #split_r = vcat(collect(2:2:12))
    split_r = vcat(collect(12:1:22))
    #rs = vcat(1, 42)
    rs = vcat(42)
    z = zeros(1,14)
    itr = 1
    while itr < 33
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
            Xx_train = deepcopy(M_train[:, vcat(collect(5:12), end-3)])
            Yy_train = deepcopy(M_train[:, end-2])
            Xx_val = deepcopy(M_val[:, vcat(collect(5:12), end-3)])
            Yy_val = deepcopy(M_val[:, end-2])
            Xx_test = deepcopy(M_pest[:, vcat(collect(5:12), end-1)])
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
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatio6_GBMs.csv"
CSV.write(savePath, optiSearch_df)


model = RandomForestClassifier(
      n_estimators = 100, 
      max_depth = 40, 
      min_samples_leaf = 12, 
      min_samples_split = 9, 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42, 
      class_weight= Dict(0=>0.625, 1=>2.501)
      )

#= model = GradientBoostingClassifier(
      learning_rate = 0.20, 
      n_estimators = 32, 
      max_depth = 18, 
      min_samples_leaf = 30, 
      min_samples_split = 20, 
      random_state = 42
      ) =#
model = GradientBoostingClassifier(
      learning_rate = 0.30, 
      n_estimators = 16, 
      max_depth = 17, 
      min_samples_leaf = 31, 
      min_samples_split = 16, 
      random_state = 42
      )

#fit!(model, Matrix(inputDB[:, vcat(collect(5:12), end-3)]), Vector(inputDB[:, end-2]))
fit!(model, Matrix(inputDB[:, vcat(collect(5:12), end-3)]), Vector(inputDB[:, end-2]), sample_weight=sampleW)
# --------------------------------------------------------------------------------------------------
#fit!(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-2]))
fit!(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-2]), sample_weight=sampleW)
# --------------------------------------------------------------------------------------------------
#fit!(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]), Vector(inputDBInputDB_test[:, end-2]))
fit!(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]), Vector(inputDBInputDB_test[:, end-2]), sample_weight=sampleTW)
# --------------------------------------------------------------------------------------------------
#fit!(model, Matrix(inputDBInputDB_test[:, 5:12]), Vector(inputDBInputDB_test[:, end-2]))
fit!(model, Matrix(inputDBInputDB_test[:, 5:12]), Vector(inputDBInputDB_test[:, end-2]), sample_weight=sampleTW)

# saving model
#modelSavePath = "F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio.joblib"
modelSavePath = "F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
#modelSavePath = "F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio.joblib"
modelSavePath = "F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
#modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatio.joblib"
modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib"
jl.dump(model, modelSavePath, compress = 5)
# --------------------------------------------------------------------------------------------------
#modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatio.joblib"
modelSavePath = "F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib"
jl.dump(model, modelSavePath, compress = 5)

describe((inputDB_pest))[vcat(collect(5:12), end-1), :]

#load a model
# requires python 3.11 or 3.12
#model = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio.joblib")
model = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
size(model)
# training performace, withDeltaRi vs. withoutDeltaRi
predictedTPTN_train = predict(model, Matrix(inputDB[:, vcat(collect(5:12), end-3)]))
inputDB[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 1637238 x 19 df
#savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
#model = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio.joblib")
model = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
size(model)
predictedTPTN_train = predict(model, Matrix(inputDB[:, 5:12]))
inputDB[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 1637238 x 19 df
#savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDB)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
#model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatio.joblib")
model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
size(model)
predictedTPTN_train = predict(model, Matrix(inputDBInputDB_test[:, vcat(collect(5:12), end-3)]))
inputDBInputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 2046364 x 19 df
#savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDBInputDB_test)
# --------------------------------------------------------------------------------------------------
#load a model
# requires python 3.11 or 3.12
#model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatio.joblib")
model = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
size(model)
predictedTPTN_train = predict(model, Matrix(inputDBInputDB_test[:, 5:12]))
inputDBInputDB_test[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 2046364 x 19 df
#savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDBInputDB_test)

# ==================================================================================================
#inputDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
inputDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv", DataFrame)
# RF: 1, 0.05284326408255855, 0.2298766279606488; GBMs: 1, 0.06497711389547518, 0.25490608838447776
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
# RF: 0.3612168205537992; GBMs: 0.22603427582725222
rSquare_train = rSquareDetermination(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])

# 1637238 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]))
# RF: 0.7641592498194061; GBMs: 0.7241870124213459
f1_train = f1_score(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])
# RF: 0.7576191480268382; GBMs: 0.7191308124902744
mcc_train = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-3], inputDB_withDeltaRiTPTN[:, end])

inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 1637238 x (19+1+2)
#savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDB_withDeltaRiTPTN)

describe((inputDB_withDeltaRiTPTN))[end-5:end, :]
# --------------------------------------------------------------------------------------------------
#inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv", DataFrame)
# RF: 1, 0.08733550039762088, 0.29552580326871775; GBMs: 1, 0.10721471160576532, 0.32743657646293167
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])
# RF: -0.011666264385671088; GBMs: -0.1962484859506588
rSquare_train = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])

# 1637238 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB_withoutDeltaRiTPTN[:, 5:12]))
# RF: 0.6549218446449645; GBMs: 0.6061548402730099
f1_train = f1_score(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])
# RF: 0.649228445686795; GBMs: 0.6029554612351099
mcc_train = matthews_corrcoef(inputDB_withoutDeltaRiTPTN[:, end-3], inputDB_withoutDeltaRiTPTN[:, end])

inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 1637238 x (19+1+2)
#savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDB_withoutDeltaRiTPTN)

describe((inputDB_withoutDeltaRiTPTN))[end-5:end, :]

# --------------------------------------------------------------------------------------------------
#inputWholeDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
inputWholeDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv", DataFrame)
# RF: 1, 0.05051740550556988, 0.22476077394770175; GBMs: 1, 0.060697901253149486, 0.24636944058293733
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])
# RF: 0.3875838557828927; GBMs: 0.2727764379721943
rSquare_train = rSquareDetermination(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])

# 2046364 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputWholeDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]))
# RF: 0.772252710333083; GBMs: 0.7378396518318088
f1_train = f1_score(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])
# RF: 0.7655156847434202; GBMs: 0.7323308532134074
mcc_train = matthews_corrcoef(inputWholeDB_withDeltaRiTPTN[:, end-3], inputWholeDB_withDeltaRiTPTN[:, end])

inputWholeDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputWholeDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 2046364 x (19+1+2)
#savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputWholeDB_withDeltaRiTPTN)

describe((inputWholeDB_withDeltaRiTPTN))[end-5:end, :]

# --------------------------------------------------------------------------------------------------
#inputWholeDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
inputWholeDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv", DataFrame)
# RF: 1, 0.08626226810088528, 0.2937043889710967; GBMs: 1, 0.10628949688325244, 0.32602070008398615
maxAE_train, MSE_train, RMSE_train = errorDetermination(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])
# RF: -0.0016453208001689923; GBMs: -0.18947072205169913
rSquare_train = rSquareDetermination(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])

# 2046364 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputWholeDB_withoutDeltaRiTPTN[:, 5:12]))
# RF: 0.65750894430087; GBMs: 0.6076720640835785
f1_train = f1_score(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])
# RF: 0.6515954481290722; GBMs: 0.6040523652743675
mcc_train = matthews_corrcoef(inputWholeDB_withoutDeltaRiTPTN[:, end-3], inputWholeDB_withoutDeltaRiTPTN[:, end])

inputWholeDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
inputWholeDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
# save, ouputing trainSet df 2046364 x (19+1+2)
#savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_WholeDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputWholeDB_withoutDeltaRiTPTN)

describe((inputWholeDB_withoutDeltaRiTPTN))[end-5:end, :]


# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
#modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio.joblib")
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
#modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio.joblib")
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
#modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
#size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
#modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
#size(modelRF_TPTN)

# ==================================================================================================

describe((inputDB_test))[end-5:end, :]

predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-3)]))
inputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 409126 x 19 df
#savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDB_test)
# --------------------------------------------------------------------------------------------------
predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]))
inputDB_test[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 409126 x 19 df
#savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDB_test)
# ==================================================================================================
#inputTestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
inputTestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv", DataFrame)
describe((inputTestDB_withDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.07676852607754091, 0.2770713375243656; GBMs: 1, 0.07534109296402575, 0.2744833200105714
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputTestDB_withDeltaRiTPTN[:, end-3], inputTestDB_withDeltaRiTPTN[:, end])
# RF: 0.07142478238515282; GBMs: 0.1036552914911415
rSquare_val = rSquareDetermination(inputTestDB_withDeltaRiTPTN[:, end-3], inputTestDB_withDeltaRiTPTN[:, end])

# 409126 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputTestDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-4)]))
# RF: 0.6586828950228212; GBMs: 0.6826978506135223
f1_test = f1_score(inputTestDB_withDeltaRiTPTN[:, end-3], inputTestDB_withDeltaRiTPTN[:, end])
# RF: 0.6368410426078788; GBMs: 0.6721799939787281
mcc_test = matthews_corrcoef(inputTestDB_withDeltaRiTPTN[:, end-3], inputTestDB_withDeltaRiTPTN[:, end])

inputTestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputTestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 62008 x (19+1+2)
#savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputTestDB_withDeltaRiTPTN)

describe((inputTestDB_withDeltaRiTPTN))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
#inputTestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
inputTestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv", DataFrame)
describe((inputTestDB_withoutDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.11229547865449764, 0.3351051755113574; GBMs: 1, 0.12331653329292198, 0.35116453877480563
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputTestDB_withoutDeltaRiTPTN[:, end-3], inputTestDB_withoutDeltaRiTPTN[:, end])
# RF: -0.30095529518099173; GBMs: -0.37572079970665495
rSquare_val = rSquareDetermination(inputTestDB_withoutDeltaRiTPTN[:, end-3], inputTestDB_withoutDeltaRiTPTN[:, end])

# 409126 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputTestDB_withoutDeltaRiTPTN[:, 5:12]))
# RF: 0.5574190565183466; GBMs: 0.5479130450366494
f1_test = f1_score(inputTestDB_withoutDeltaRiTPTN[:, end-3], inputTestDB_withoutDeltaRiTPTN[:, end])
# RF: 0.532787139631841; GBMs: 0.5316679181258175
mcc_test = matthews_corrcoef(inputTestDB_withoutDeltaRiTPTN[:, end-3], inputTestDB_withoutDeltaRiTPTN[:, end])

inputTestDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputTestDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 409126 x 19+2 df 
#savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputTestDB_withoutDeltaRiTPTN)

describe((inputTestDB_withoutDeltaRiTPTN))[end-4:end, :]

# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
#modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio.joblib")
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
#modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio.joblib")
modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_withoutAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
#modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
#size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
#modelRF_TPTN = jl.load("F:\\UvA\\modelTPTNModeling_WholeWithoutAbsDeltaRi_0d5FinalScoreRatio_GBMs.joblib")
#size(modelRF_TPTN)

# ==================================================================================================

describe((inputDB_pest))[end-5:end, :]

predictedTPTN_pest = predict(modelRF_TPTN, Matrix(inputDB_pest[:, vcat(collect(5:12), end-1)]))
inputDB_pest[!, "withDeltaRipredictTPTN"] = predictedTPTN_pest
# save, ouputing testSet df 62008 x 19 df
#savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDB_pest)
# --------------------------------------------------------------------------------------------------
predictedTPTN_pest = predict(modelRF_TPTN, Matrix(inputDB_pest[:, 5:12]))
inputDB_pest[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_pest
# save, ouputing testSet df 62008 x 19 df
#savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputDB_pest)
# ==================================================================================================
#inputPestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
inputPestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv", DataFrame)
describe((inputPestDB_withDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.06713649851632048, 0.25910711784186957; GBMs: 1, 0.08157011998451813, 0.2856048318647955
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])
# RF: 0.4511046480696742; GBMs: 0.33344455467484746
rSquare_val = rSquareDetermination(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])

# 62008 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withDeltaRiTPTN[:, vcat(collect(5:12), end-2)]))
# RF: 0.7664516129032258; GBMs: 0.7055878928987195
f1_test = f1_score(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])
# RF: 0.7272788279812913; GBMs: 0.6586872613324458
mcc_test = matthews_corrcoef(inputPestDB_withDeltaRiTPTN[:, end-1], inputPestDB_withDeltaRiTPTN[:, end])

inputPestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 62008 x (19+1+2)
#savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputPestDB_withDeltaRiTPTN)

describe((inputPestDB_withDeltaRiTPTN))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
#inputPestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio.csv", DataFrame)
inputPestDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTN_0d5FinalScoreRatio_GBMs.csv", DataFrame)
describe((inputPestDB_withoutDeltaRiTPTN))[end-5:end, :]

# RF: 1, 0.06426590117404206, 0.2535072014244212; GBMs: 1, 0.07555476712682234, 0.2748722742053522
maxAE_val, MSE_val, RMSE_val = errorDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])
# RF: 0.47668278565308986; GBMs: 0.3822670910210907
rSquare_val = rSquareDetermination(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])

# 62008 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputPestDB_withoutDeltaRiTPTN[:, 5:12]))
# RF: 0.7910878112712975; GBMs: 0.7337614366085129
f1_test = f1_score(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])
# RF: 0.75612508340891; GBMs: 0.6897524889855885
mcc_test = matthews_corrcoef(inputPestDB_withoutDeltaRiTPTN[:, end-1], inputPestDB_withoutDeltaRiTPTN[:, end])

inputPestDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
inputPestDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
# save, ouputing trainSet df 62008 x 19+2 df 
#savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_PestDF_withoutAbsDeltaRiandPredictedTPTNandpTP_0d5FinalScoreRatio_GBMs.csv"
CSV.write(savePath, inputPestDB_withoutDeltaRiTPTN)

describe((inputPestDB_withoutDeltaRiTPTN))[end-4:end, :]
