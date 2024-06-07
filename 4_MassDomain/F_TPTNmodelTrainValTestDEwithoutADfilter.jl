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
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDF.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
insertcols!(inputDB_test, 10, ("MatchRatio"=>float(0)))
inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
for i = 1:size(inputDB_test, 1)
    inputDB_test[i, "RefMatchFragRatio"] = log10(inputDB_test[i, "RefMatchFragRatio"])
    inputDB_test[i, "UsrMatchFragRatio"] = log10(inputDB_test[i, "UsrMatchFragRatio"])
    inputDB_test[i, "FinalScoreRatio"] = log10(inputDB_test[i, "FinalScoreRatio"])
    inputDB_test[i, "MatchRatio"] = inputDB_test[i, "DirectMatch"] - inputDB_test[i, "ReversMatch"]
end
# save, ouputing 409126 x 19 df, 0:373107; 1:36019 = 0.5483; 5.6793
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF0d5FinalScoreRatioDE.csv"
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
end
# save, ouputing 1637238 x 19 df, 0:1492750; 1:144488 = 0.5484; 5.6657
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB)
inputDB[inputDB.LABEL .== 1, :]

# 4103848 x 19 df; 
# 409126+1637238= 2046364, 0:1865857; 1:180507 = 0.5484; 5.6684
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
savePath = "F:\\UvA\\dataframeTPTNModeling_pestDF0d5FinalScoreRatioDE.csv"
CSV.write(savePath, inputDB_pest)
inputDB_pest[inputDB_pest.LABEL .== 0, :]

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
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatioDE_RFnewCompare1.csv"
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
    lr_r = vcat(collect(0.1:0.1:0.5))
    leaf_r = vcat(collect(16:2:32))
    tree_r = vcat(8, 16, 24, 32)
    depth_r = vcat(collect(10:2:20))
    split_r = vcat(collect(12:2:20))
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
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatioDE_GBMsnew1.csv"
CSV.write(savePath, optiSearch_df)
