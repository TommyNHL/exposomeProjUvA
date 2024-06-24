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

# inputing 485631 x 22 df
# 0: 334321; 1: 151310 = 0.7263; 1.6048
trainDEFSDf = CSV.read("F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilterSTD.csv", DataFrame)
trainDEFSDf[trainDEFSDf.LABEL .== 1, :]
describe(trainDEFSDf)

Yy_train = deepcopy(trainDEFSDf[:, end-4])  # 0.7263; 1.6048
sampleW = []
for w in Vector(Yy_train)
    if w == 0
        push!(sampleW, 0.7263)
    elseif w == 1
        push!(sampleW, 1.6048)
    end
end 

# inputing 121946 x 22 df
# 0: 83981; 1: 37965 = 0.7260; 1.6060
testDEFSDf = CSV.read("F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilterSTD.csv", DataFrame)
testDEFSDf[testDEFSDf.LABEL .== 1, :]

Yy_val = deepcopy(testDEFSDf[:, end-4])  # 0.7260; 1.6060
sampletestW = []
for w in Vector(Yy_val)
    if w == 0
        push!(sampletestW, 0.7260)
    elseif w == 1
        push!(sampletestW, 1.6060)
    end
end 

# 607577 x 22 df; 
# 485631+121946= 607577, 0:418302; 1:189275 = 
wholeDEFSDf = vcat(trainDEFSDf, testDEFSDf)
sort!(wholeDEFSDf, [:ENTRY])
wholeDEFSDf[wholeDEFSDf.LABEL .== 1, :]


# 10868 x 19 df
# 0: 7133; 1: 3735 = 0.7618; 1.4549
noTeaDEFSDf = CSV.read("F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilterSTD.csv", DataFrame)
noTeaDEFSDf[noTeaDEFSDf.LABEL .== 1, :]

Yy_test = deepcopy(noTeaDEFSDf[:, end-1])  #  0.7618; 1.4549
samplepestW = []
for w in Vector(Yy_test)
    if w == 0
        push!(samplepestW, 0.7618)
    elseif w == 1
        push!(samplepestW, 1.4549)
    end
end 

# 29397 x 19 df
# 1: 8187
TeaDEFSDf = CSV.read("F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilterSTD.csv", DataFrame)
TeaDEFSDf = TeaDEFSDf[TeaDEFSDf.LABEL .== 1, :]

Yy_test2 = deepcopy(TeaDEFSDf[:, end-1])  #  0.7618; 1.4549
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
@sk_import neural_network : MLPClassifier
@sk_import svm : SVC
@sk_import neighbors : KNeighborsClassifier
@sk_import metrics : recall_score
@sk_import inspection  : permutation_importance
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
describe((trainDEFSDf))[vcat(5,6,9,10,13,14, end-5), :]
describe((testDEFSDf))[vcat(5,6,9,10,13,14, end-5), :]
describe((noTeaDEFSDf))[vcat(5,6,9,10,13,14, end-2), :]
describe((TeaDEFSDf))[vcat(5,6,9,10,13,14, end-2), :]
describe((TeaDEFSDf))
#-------------------------------------------------------------------------------
function rankRandomForestClass(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    leaf_r = 5  # 3
    depth_r = 185 # 3
    split_r = 52  # 5
    rs = 42
    z = zeros(1,14)
    itr = 1
    t = 150
    mod = 0
    rank = vcat(5,6,7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    for l in leaf_r
        for d in depth_r
            for r in split_r
                if itr == 1
                    println("itr=", itr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
                    println("## loading in data ##")
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
                    println("## Classification ##")
                    reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, n_jobs=-1, oob_score =true, random_state=rs, class_weight=Dict(0=>0.9526, 1=>1.0524))  # 0.7263; 1.6048
                    println("## fit ##")
                    fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                    importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42, n_jobs=-1)
                    return importances
                end
            end
        end
    end
end

rank = rankRandomForestClass(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)
#-------------------------------------------------------------------------------
rank
#-------------------------------------------------------------------------------

function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    leaf_r = vcat(collect(50:25:100))  # 3
    depth_r = vcat(collect(2:20:62))  # 3
    split_r = vcat(collect(2:20:102))  # 5
    rs = 42
    z = zeros(1,14)
    itr = 1
    t = 50
    mod = 0
    rank = vcat(5,6,7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    for l in leaf_r
        for d in depth_r
            for r in split_r
                if itr == 1
                    println("itr=", itr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
                    println("## loading in data ##")
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
                    println("## Classification ##")
                    reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, n_jobs=-1, oob_score =true, random_state=rs, class_weight=Dict(0=>0.9526, 1=>1.0524))  # 0.7263; 1.6048
                    println("## fit ##")
                    fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                    importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42, n_jobs=-1)
                    importances["importances_mean", :]
                    print(importances)
                    z[1,1] = l
                    z[1,2] = t
                    z[1,3] = d
                    z[1,4] = r
                    z[1,5] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,6] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,7] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    z[1,8] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    z[1,9] = avgScore(f1_10_train, 3)
                    z[1,10] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,11] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,12] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    z[1,13] = rs
                    z[1,14] = mod
                    println(z[end, :])
                else
                    println("itr=", itr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
                    println("## loading in data ##")
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
                    println("## Classification ##")
                    reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, n_jobs=-1, oob_score =true, random_state=rs, class_weight=Dict(0=>0.9526, 1=>1.0524))  # 0.7263; 1.6048
                    println("## fit ##")
                    fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                    
                    itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    traincvtrain = avgScore(f1_10_train, 3) 
                    f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    rec = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    z = vcat(z, [l t d r itrain jtrain ival jval traincvtrain f1s mccs rec rs mod])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], depth = z[:,3], minSplit = z[:,4], f1_train = z[:,5], mcc_train = z[:,6], f1_val = z[:,7], mcc_val = z[:,8], f1_3Ftrain = z[:,9], f1_pest = z[:,10], mcc_pest = z[:,11], recall = z[:,12], state = z[:,13], model = z[:,14])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestClass(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_RF2.csv"
CSV.write(savePath, optiSearch_df)

#===============================================================================#

model = RandomForestClassifier(
      n_estimators = 425, 
      max_depth = 100, 
      min_samples_leaf = 28, 
      #max_features = Int64(9), 
      min_samples_split = 12, 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42, 
      class_weight= Dict(0=>0.7183, 1=>1.6453)  # 0.7183; 1.6453
      )
