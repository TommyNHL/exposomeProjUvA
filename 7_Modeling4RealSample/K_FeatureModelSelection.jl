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
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    leaf_r = 2  # 1
    depth_r = vcat(collect(2:1:7))  # 6
    split_r = 2  # 1
    l = 2
    r = 2
    rs = 42
    z = zeros(1,28)
    itr = 1
    tree_r = vcat(collect(50:50:500))  # 9
    mod = 0
    rank = vcat(5, 7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    for l in leaf_r
        for d in depth_r
            for t in tree_r
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
                reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, n_jobs=-1, oob_score =true, random_state=rs, class_weight=Dict(0=>0.9929, 1=>1.0072))  # 0.7263; 1.6048
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42, n_jobs=-1)
                print(importances["importances_mean"])
                if itr == 1
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
                    z[1,15] = importances["importances_mean"][1]
                    z[1,16] = importances["importances_mean"][2]
                    z[1,17] = importances["importances_mean"][3]
                    z[1,18] = importances["importances_mean"][4]
                    z[1,19] = importances["importances_mean"][5]
                    z[1,20] = importances["importances_mean"][6]
                    z[1,21] = importances["importances_mean"][7]
                    z[1,22] = importances["importances_std"][1]
                    z[1,23] = importances["importances_std"][2]
                    z[1,24] = importances["importances_std"][3]
                    z[1,25] = importances["importances_std"][4]
                    z[1,26] = importances["importances_std"][5]
                    z[1,27] = importances["importances_std"][6]
                    z[1,28] = importances["importances_std"][7]
                    println(z[end, :])
                else
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
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    z = vcat(z, [l t d r itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 sd1 sd2 sd3 sd4 sd5 sd6 sd7])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], depth = z[:,3], minSplit = z[:,4], f1_train = z[:,5], mcc_train = z[:,6], f1_val = z[:,7], mcc_val = z[:,8], f1_3Ftrain = z[:,9], f1_pest = z[:,10], mcc_pest = z[:,11], recall = z[:,12], state = z[:,13], model = z[:,14], im1 = z[:,15], im2 = z[:,16], im3 = z[:,17], im4 = z[:,18], im5 = z[:,19], im6 = z[:,20], im7 = z[:,21], sd1 = z[:,22], sd2 = z[:,23], sd3 = z[:,24], sd4 = z[:,25], sd5 = z[:,26], sd6 = z[:,27], sd7 = z[:,28])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestClass(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_RF6_noFilterLog(UsrFragMatchRatio).csv"
CSV.write(savePath, optiSearch_df)


function optimRandomForestClass2(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    leaf_r = 2  # 1
    depth_r = vcat(collect(2:1:7))  # 6
    split_r = 2  # 1
    l = 2
    r = 2
    rs = 42
    z = zeros(1,28)
    itr = 1
    tree_r = vcat(collect(50:50:500))  # 9
    mod = 0
    rank = vcat(5, 7,9,10,13,14, 17)
    M_train = inputDB
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    for l in leaf_r
        for d in depth_r
            for t in tree_r
                println("itr=", itr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
                println("## loading in data ##")
                Xx_train = deepcopy(M_train[:, rank])
                Xx_val = deepcopy(M_val[:, rank])
                Xx_test = deepcopy(M_pest[:, rank])
                Xx_test2 = deepcopy(M_pest2[:, rank])
                #
                Yy_train = deepcopy(M_train[:, end-4])
                Yy_val = deepcopy(M_val[:, end-4])
                Yy_test = deepcopy(M_pest[:, end-1])
                Yy_test2 = deepcopy(M_pest2[:, end-1])
                println("## Classification ##")
                reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, n_jobs=-1, oob_score =true, random_state=rs, class_weight=Dict(0=>0.5493, 1=>5.5724))  # 0.5493; 5.5724
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42, n_jobs=-1)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = l
                    z[1,2] = t
                    z[1,3] = d
                    z[1,4] = r
                    z[1,5] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                    z[1,6] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                    z[1,7] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    z[1,8] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)  # imbalance
                    z[1,9] = avgScore(f1_10_train, 3)
                    z[1,10] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,11] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,12] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    z[1,13] = rs
                    z[1,14] = mod
                    z[1,15] = importances["importances_mean"][1]
                    z[1,16] = importances["importances_mean"][2]
                    z[1,17] = importances["importances_mean"][3]
                    z[1,18] = importances["importances_mean"][4]
                    z[1,19] = importances["importances_mean"][5]
                    z[1,20] = importances["importances_mean"][6]
                    z[1,21] = importances["importances_mean"][7]
                    z[1,22] = importances["importances_std"][1]
                    z[1,23] = importances["importances_std"][2]
                    z[1,24] = importances["importances_std"][3]
                    z[1,25] = importances["importances_std"][4]
                    z[1,26] = importances["importances_std"][5]
                    z[1,27] = importances["importances_std"][6]
                    z[1,28] = importances["importances_std"][7]
                    println(z[end, :])
                else
                    itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                    jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                    ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    traincvtrain = avgScore(f1_10_train, 3) 
                    f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    rec = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    z = vcat(z, [l t d r itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 sd1 sd2 sd3 sd4 sd5 sd6 sd7])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], depth = z[:,3], minSplit = z[:,4], f1_train = z[:,5], mcc_train = z[:,6], f1_val = z[:,7], mcc_val = z[:,8], f1_3Ftrain = z[:,9], f1_pest = z[:,10], mcc_pest = z[:,11], recall = z[:,12], state = z[:,13], model = z[:,14], im1 = z[:,15], im2 = z[:,16], im3 = z[:,17], im4 = z[:,18], im5 = z[:,19], im6 = z[:,20], im7 = z[:,21], sd1 = z[:,22], sd2 = z[:,23], sd3 = z[:,24], sd4 = z[:,25], sd5 = z[:,26], sd6 = z[:,27], sd7 = z[:,28])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df2 = optimRandomForestClass2(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_RF7imbalance_noFilterLog(UsrFragMatchRatio).csv"
CSV.write(savePath, optiSearch_df2)
#-------------------------------------------------------------------------------

function optimDecisionTreeClass(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    leaf_r = vcat(collect(2:250:2002))  # 9
    depth_r = vcat(collect(2:1:5))  # 4
    split_r = vcat(collect(2:1:10))  # 9
    rs = 42
    z = zeros(1,27)
    itr = 1
    mod = 0
    rank = vcat(5, 7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    for l in leaf_r
        for d in depth_r
            for r in split_r
                println("itr=", itr, ", leaf=", l, ", depth=", d, ", minSsplit=", r, ", model=", mod)
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
                reg = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, min_samples_split=r, random_state=rs, class_weight=Dict(0=>0.9929, 1=>1.0072))  # 0.7263; 1.6048
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = l
                    z[1,2] = d
                    z[1,3] = r
                    z[1,4] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,5] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,6] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    z[1,7] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    z[1,8] = avgScore(f1_10_train, 3)
                    z[1,9] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,10] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,11] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    z[1,12] = rs
                    z[1,13] = mod
                    z[1,14] = importances["importances_mean"][1]
                    z[1,15] = importances["importances_mean"][2]
                    z[1,16] = importances["importances_mean"][3]
                    z[1,17] = importances["importances_mean"][4]
                    z[1,18] = importances["importances_mean"][5]
                    z[1,19] = importances["importances_mean"][6]
                    z[1,20] = importances["importances_mean"][7]
                    z[1,21] = importances["importances_std"][1]
                    z[1,22] = importances["importances_std"][2]
                    z[1,23] = importances["importances_std"][3]
                    z[1,24] = importances["importances_std"][4]
                    z[1,25] = importances["importances_std"][5]
                    z[1,26] = importances["importances_std"][6]
                    z[1,27] = importances["importances_std"][7]
                    println(z[end, :])
                else
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
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    z = vcat(z, [l d r itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 sd1 sd2 sd3 sd4 sd5 sd6 sd7])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], depth = z[:,2], minSplit = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_val = z[:,6], mcc_val = z[:,7], f1_3Ftrain = z[:,8], f1_pest = z[:,9], mcc_pest = z[:,10], recall = z[:,11], state = z[:,12], model = z[:,13], im1 = z[:,14], im2 = z[:,15], im3 = z[:,16], im4 = z[:,17], im5 = z[:,18], im6 = z[:,19], im7 = z[:,20], sd1 = z[:,21], sd2 = z[:,22], sd3 = z[:,23], sd4 = z[:,24], sd5 = z[:,25], sd6 = z[:,26], sd7 = z[:,27])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimDecisionTreeClass(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_DT2_noFilterUsrFragMatchRatio.csv"
CSV.write(savePath, optiSearch_df)

#-------------------------------------------------------------------------------

function optimKNN(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    k_n_r = vcat(collect(295:10:395))  # 11
    leaf_r = vcat(300, 400)  # 2
    w_r = ["uniform", "distance"]
    met_r = ["minkowski", "euclidean", "manhattan"]
    p_r = vcat(1, 2)
    z = zeros(1,26)
    w = 1
    mod = 0
    itr = 1
    met = 1
    p = 2
    leaf = 300
    rank = vcat(5, 7,9, 13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    for k_n in k_n_r
        #or leaf in leaf_r
            #for w in 1:2
            #for met in vcat(1,3)
                #for p in p_r
        println("k_n=", k_n, ", leaf=", leaf, ", w=", w_r[w], ", met=", met_r[met], ", p=", p, ", model=", mod)
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
        reg = KNeighborsClassifier(n_neighbors=k_n, weights=w_r[w], leaf_size=leaf, p=p, metric=met_r[met])  # 0.7263; 1.6048
        println("## fit ##")
        fit!(reg, Matrix(Xx_train), Vector(Yy_train))
        importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
        print(importances["importances_mean"])
        if itr == 1
            z[1,1] = k_n
            z[1,2] = leaf
            z[1,3] = w
            z[1,4] = met
            z[1,5] = p
            z[1,6] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
            z[1,7] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
            z[1,8] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
            z[1,9] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
            println("## CV ##")
            f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
            z[1,10] = avgScore(f1_10_train, 3)
            z[1,11] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
            z[1,12] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
            z[1,13] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
            z[1,14] = mod
            z[1,15] = importances["importances_mean"][1]
            z[1,16] = importances["importances_mean"][2]
            z[1,17] = importances["importances_mean"][3]
            z[1,18] = importances["importances_mean"][4]
            z[1,19] = importances["importances_mean"][5]
            z[1,20] = importances["importances_mean"][6]
            z[1,21] = importances["importances_std"][1]
            z[1,22] = importances["importances_std"][2]
            z[1,23] = importances["importances_std"][3]
            z[1,24] = importances["importances_std"][4]
            z[1,25] = importances["importances_std"][5]
            z[1,26] = importances["importances_std"][6]
            println(z[end, :])
        else
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
            im1 = importances["importances_mean"][1]
            im2 = importances["importances_mean"][2]
            im3 = importances["importances_mean"][3]
            im4 = importances["importances_mean"][4]
            im5 = importances["importances_mean"][5]
            im6 = importances["importances_mean"][6]
            sd1 = importances["importances_std"][1]
            sd2 = importances["importances_std"][2]
            sd3 = importances["importances_std"][3]
            sd4 = importances["importances_std"][4]
            sd5 = importances["importances_std"][5]
            sd6 = importances["importances_std"][6]
            z = vcat(z, [k_n leaf w met p itrain jtrain ival jval traincvtrain f1s mccs rec mod im1 im2 im3 im4 im5 im6 sd1 sd2 sd3 sd4 sd5 sd6])
            println(z[end, :])
        end
        println("End of ", itr, " iterations")
        itr += 1
        #end
                #end
            #end
        #end
    end
    z_df = DataFrame(k_n = z[:,1], leaf = z[:,2], weight = z[:,3], met = z[:,4], p = z[:,5], f1_train = z[:,6], mcc_train = z[:,7], f1_val = z[:,8], mcc_val = z[:,9], f1_3Ftrain = z[:,10], f1_pest = z[:,11], mcc_pest = z[:,12], recall = z[:,13], model = z[:,14], im1 = z[:,15], im2 = z[:,16], im3 = z[:,17], im4 = z[:,18], im5 = z[:,19], im6 = z[:,20], sd1 = z[:,21], sd2 = z[:,22], sd3 = z[:,23], sd4 = z[:,24], sd5 = z[:,25], sd6 = z[:,26])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimKNN(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\F\\UvA\\app\\hyperparameterTuning_modelSelection_KNN16_noMatchDiff.csv"
CSV.write(savePath, optiSearch_df)

#-------------------------------------------------------------------------------

function optimLR(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    penalty_r = ["l1", "l2"]
    solver_rs = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    c_values_r = vcat(1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001)
    rs = 42
    itr = 1
    mod = 0
    rank = vcat(5, 7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    pnn = 0
    z = zeros(1,27)
    for pn in 1:2
        for s in 1:5
            for c in c_values_r
                if solver_rs[s] == "lbfgs" || solver_rs[s] == "newton-cg" || solver_rs[s] == "sag"
                    pnn = 2
                else
                    pnn = pn
                end
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
                reg = LogisticRegression(penalty=penalty_r[pnn], C=c, solver=solver_rs[s], max_iter=5000, random_state=rs, class_weight=Dict(0=>0.9929, 1=>1.0072))  # 0.7263; 1.6048
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = pnn
                    z[1,2] = s
                    z[1,3] = c
                    z[1,4] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,5] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,6] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    z[1,7] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    z[1,8] = avgScore(f1_10_train, 3)
                    z[1,9] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,10] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,11] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    z[1,12] = rs
                    z[1,13] = mod
                    z[1,14] = importances["importances_mean"][1]
                    z[1,15] = importances["importances_mean"][2]
                    z[1,16] = importances["importances_mean"][3]
                    z[1,17] = importances["importances_mean"][4]
                    z[1,18] = importances["importances_mean"][5]
                    z[1,19] = importances["importances_mean"][6]
                    z[1,20] = importances["importances_mean"][7]
                    z[1,21] = importances["importances_std"][1]
                    z[1,22] = importances["importances_std"][2]
                    z[1,23] = importances["importances_std"][3]
                    z[1,24] = importances["importances_std"][4]
                    z[1,25] = importances["importances_std"][5]
                    z[1,26] = importances["importances_std"][6]
                    z[1,27] = importances["importances_std"][7]
                    println(z[end, :])
                else
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
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    z = vcat(z, [pnn s c itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 sd1 sd2 sd3 sd4 sd5 sd6 sd7])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(Penalty = z[:,1], Solver = z[:,2], C_value = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_val = z[:,6], mcc_val = z[:,7], f1_3Ftrain = z[:,8], f1_pest = z[:,9], mcc_pest = z[:,10], recall = z[:,11], state = z[:,12], model = z[:,13], im1 = z[:,14], im2 = z[:,15], im3 = z[:,16], im4 = z[:,17], im5 = z[:,18], im6 = z[:,19], im7 = z[:,20], sd1 = z[:,21], sd2 = z[:,22], sd3 = z[:,23], sd4 = z[:,24], sd5 = z[:,25], sd6 = z[:,26], sd7 = z[:,27])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimLR(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_LR4_noFilterLog(UsrFragMatchRatio).csv"
CSV.write(savePath, optiSearch_df)

#-------------------------------------------------------------------------------

function optimSVM(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    c_values_r = vcat(1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001)  # 6
    kernel_r = ["linear", "poly", "rbf", "sigmoid"]  # 4
    gamma_r = ["scale", "auto"] # 2
    rs = 42
    z = zeros(1,29)
    mod = 0
    rank = vcat(5,6,7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    itr = 1
    for c in c_values_r
        for k in 1:4
            for g in 1:2
                println("itr=", itr, ",C=", c, ", K=", k, ", G=", g, ", model=", mod)
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
                reg = SVC(C=c, kernel=kernel_r[k], gamma=gamma_r[g], random_state=rs, class_weight=Dict(0=>0.9929, 1=>1.0072))
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = c
                    z[1,2] = k
                    z[1,3] = g
                    z[1,4] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,5] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,6] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    z[1,7] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    z[1,8] = avgScore(f1_10_train, 3)
                    z[1,9] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,10] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,11] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    z[1,12] = rs
                    z[1,13] = mod
                    z[1,14] = importances["importances_mean"][1]
                    z[1,15] = importances["importances_mean"][2]
                    z[1,16] = importances["importances_mean"][3]
                    z[1,17] = importances["importances_mean"][4]
                    z[1,18] = importances["importances_mean"][5]
                    z[1,19] = importances["importances_mean"][6]
                    z[1,20] = importances["importances_mean"][7]
                    z[1,21] = importances["importances_mean"][8]
                    z[1,22] = importances["importances_std"][1]
                    z[1,23] = importances["importances_std"][2]
                    z[1,24] = importances["importances_std"][3]
                    z[1,25] = importances["importances_std"][4]
                    z[1,26] = importances["importances_std"][5]
                    z[1,27] = importances["importances_std"][6]
                    z[1,28] = importances["importances_std"][7]
                    z[1,29] = importances["importances_std"][8]
                    println(z[end, :])
                else
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
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    im8 = importances["importances_mean"][8]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    sd8 = importances["importances_std"][8]
                    z = vcat(z, [c k g itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 im8 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(C = z[:,1], kernel = z[:,2], gamma = z[:,3], f1_train = z[:,6], mcc_train = z[:,7], f1_val = z[:,8], mcc_val = z[:,9], f1_3Ftrain = z[:,10], f1_pest = z[:,11], mcc_pest = z[:,12], recall = z[:,13], state = z[:,14], model = z[:,15], im1 = z[:,16], im2 = z[:,17], im3 = z[:,18], im4 = z[:,19], im5 = z[:,20], im6 = z[:,21], im7 = z[:,22], im8 = z[:,23], sd1 = z[:,24], sd2 = z[:,25], sd3 = z[:,26], sd4 = z[:,27], sd5 = z[:,28], sd6 = z[:,29], sd7 = z[:,30], sd8 = z[:,31])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimSVM(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_SVM1_noFilter.csv"
CSV.write(savePath, optiSearch_df)

#-------------------------------------------------------------------------------

function optimGradientBoostClass(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    lr_r = vcat(1, 0.1, 0.01, 0.001)  # 4
    leaf_r = vcat(1,2)
    tree_r = vcat(collect(50:100:350))  # 4
    depth_r = vcat(collect(2:1:6))  # 5
    split_r = vcat(2)
    rs = 42
    z = zeros(1,31)
    l = 2
    r = 2
    mod = 0
    rank = vcat(5,6,7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    itr = 1
    for lr in lr_r
        for t in tree_r
            for d in depth_r
                println("itr=", itr, ", lr=", lr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
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
                reg = GradientBoostingClassifier(learning_rate=lr, n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, random_state=rs)
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = lr
                    z[1,2] = l
                    z[1,3] = t
                    z[1,4] = d
                    z[1,5] = r
                    z[1,6] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,7] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,8] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    z[1,9] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    z[1,10] = avgScore(f1_10_train, 3)
                    z[1,11] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,12] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,13] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    z[1,14] = rs
                    z[1,15] = mod
                    z[1,16] = importances["importances_mean"][1]
                    z[1,17] = importances["importances_mean"][2]
                    z[1,18] = importances["importances_mean"][3]
                    z[1,19] = importances["importances_mean"][4]
                    z[1,20] = importances["importances_mean"][5]
                    z[1,21] = importances["importances_mean"][6]
                    z[1,22] = importances["importances_mean"][7]
                    z[1,23] = importances["importances_mean"][8]
                    z[1,24] = importances["importances_std"][1]
                    z[1,25] = importances["importances_std"][2]
                    z[1,26] = importances["importances_std"][3]
                    z[1,27] = importances["importances_std"][4]
                    z[1,28] = importances["importances_std"][5]
                    z[1,29] = importances["importances_std"][6]
                    z[1,30] = importances["importances_std"][7]
                    z[1,31] = importances["importances_std"][8]
                    println(z[end, :])
                else
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
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    im8 = importances["importances_mean"][8]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    sd8 = importances["importances_std"][8]
                    z = vcat(z, [lr l t d r itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 im8 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(learnRate = z[:,1], leaves = z[:,2], trees = z[:,3], depth = z[:,4], minSplit = z[:,5], f1_train = z[:,6], mcc_train = z[:,7], f1_val = z[:,8], mcc_val = z[:,9], f1_3Ftrain = z[:,10], f1_pest = z[:,11], mcc_pest = z[:,12], recall = z[:,13], state = z[:,14], model = z[:,15], im1 = z[:,16], im2 = z[:,17], im3 = z[:,18], im4 = z[:,19], im5 = z[:,20], im6 = z[:,21], im7 = z[:,22], im8 = z[:,23], sd1 = z[:,24], sd2 = z[:,25], sd3 = z[:,26], sd4 = z[:,27], sd5 = z[:,28], sd6 = z[:,29], sd7 = z[:,30], sd8 = z[:,31])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimGradientBoostClass(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_GBM1_noFilter.csv"
CSV.write(savePath, optiSearch_df)

#-------------------------------------------------------------------------------

function optimAdaBoostClass(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    lr_r = vcat(1, 0.1, 0.01, 0.001)  # 4
    leaf_r = vcat(1,2)
    tree_r = vcat(collect(50:100:350))  # 4
    depth_r = vcat(collect(2:1:6))  # 5
    split_r = vcat(2)
    rs = 42
    z = zeros(1,31)
    l = 2
    r = 2
    mod = 0
    rank = vcat(5,6,7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    itr = 1
    for lr in lr_r
        for t in tree_r
            for d in depth_r
                dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, min_samples_split=r, random_state=rs, class_weight=Dict(0=>0.9929, 1=>1.0072))
                println("itr=", itr, ", lr=", lr, ", tree=", t, ", model=", mod)
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
                reg = AdaBoostClassifier(estimator=dtc, n_estimators=t, algorithm="SAMME", learning_rate=lr, random_state=rs)
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = lr
                    z[1,2] = l
                    z[1,3] = t
                    z[1,4] = d
                    z[1,5] = r
                    z[1,6] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,7] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,8] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    z[1,9] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    z[1,10] = avgScore(f1_10_train, 3)
                    z[1,11] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,12] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,13] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    z[1,14] = rs
                    z[1,15] = mod
                    z[1,16] = importances["importances_mean"][1]
                    z[1,17] = importances["importances_mean"][2]
                    z[1,18] = importances["importances_mean"][3]
                    z[1,19] = importances["importances_mean"][4]
                    z[1,20] = importances["importances_mean"][5]
                    z[1,21] = importances["importances_mean"][6]
                    z[1,22] = importances["importances_mean"][7]
                    z[1,23] = importances["importances_mean"][8]
                    z[1,24] = importances["importances_std"][1]
                    z[1,25] = importances["importances_std"][2]
                    z[1,26] = importances["importances_std"][3]
                    z[1,27] = importances["importances_std"][4]
                    z[1,28] = importances["importances_std"][5]
                    z[1,29] = importances["importances_std"][6]
                    z[1,30] = importances["importances_std"][7]
                    z[1,31] = importances["importances_std"][8]
                    println(z[end, :])
                else
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
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    im8 = importances["importances_mean"][8]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    sd8 = importances["importances_std"][8]
                    z = vcat(z, [lr l t d r itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 im8 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(learnRate = z[:,1], leaves = z[:,2], trees = z[:,3], depth = z[:,4], minSplit = z[:,5], f1_train = z[:,6], mcc_train = z[:,7], f1_val = z[:,8], mcc_val = z[:,9], f1_3Ftrain = z[:,10], f1_pest = z[:,11], mcc_pest = z[:,12], recall = z[:,13], state = z[:,14], model = z[:,15], im1 = z[:,16], im2 = z[:,17], im3 = z[:,18], im4 = z[:,19], im5 = z[:,20], im6 = z[:,21], im7 = z[:,22], im8 = z[:,23], sd1 = z[:,24], sd2 = z[:,25], sd3 = z[:,26], sd4 = z[:,27], sd5 = z[:,28], sd6 = z[:,29], sd7 = z[:,30], sd8 = z[:,31])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimAdaBoostClass(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_Ada1_noFilter.csv"
CSV.write(savePath, optiSearch_df)

#-------------------------------------------------------------------------------

function optimMLP(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    hls_r = [(8,8,8), (8,16,8), (8,16,16), (16,16,16), (16,16,8), (16,8,8), (16,8,16), (8,16,16)]  # 8
    maxIter_r = vcat(100, 200)  # 2
    alpha_r = [0.0001, 0.05]  # 2
    act_r = ["tanh", "relu"]  # 2
    solver_r = ["sgd", "adam"]  # 2
    lr_r = ["constant", "adaptive"]  # 2
    rs = 42
    z = zeros(1,32)
    mod = 0
    rank = vcat(5,6,7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    itr = 1
    for hls in 1:8
        for it in maxIter_r
            for alph in alpha_r
                for act in vcat(1,2)
                    for sol in vcat(1,2)
                        for lr in vcat(1,2)
                            println("itr=", itr, ", hls=", hls, ", maxit=", it, ", act=", act, ", solver=", sol, ", alph=", alph, ", lr=", lr, ", model=", mod)
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
                            reg = MLPClassifier(hidden_layer_sizes=hls_r[hls], max_iter=it, activation=act_r[act], solver=solver_r[sol], alpha=alph, learning_rate=lr_r[lr], random_state=rs)  # 0.7263; 1.6048
                            println("## fit ##")
                            fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                            importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
                            print(importances["importances_mean"])
                            if itr == 1
                                z[1,1] = hls
                                z[1,2] = it
                                z[1,3] = alph
                                z[1,4] = act
                                z[1,5] = sol
                                z[1,6] = lr
                                z[1,7] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                                z[1,8] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                                z[1,9] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                                z[1,10] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                                println("## CV ##")
                                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                                z[1,11] = avgScore(f1_10_train, 3)
                                z[1,12] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                                z[1,13] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                                z[1,14] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                                z[1,15] = rs
                                z[1,16] = mod
                                z[1,17] = importances["importances_mean"][1]
                                z[1,18] = importances["importances_mean"][2]
                                z[1,19] = importances["importances_mean"][3]
                                z[1,20] = importances["importances_mean"][4]
                                z[1,21] = importances["importances_mean"][5]
                                z[1,22] = importances["importances_mean"][6]
                                z[1,23] = importances["importances_mean"][7]
                                z[1,24] = importances["importances_mean"][8]
                                z[1,25] = importances["importances_std"][1]
                                z[1,26] = importances["importances_std"][2]
                                z[1,27] = importances["importances_std"][3]
                                z[1,28] = importances["importances_std"][4]
                                z[1,29] = importances["importances_std"][5]
                                z[1,30] = importances["importances_std"][6]
                                z[1,31] = importances["importances_std"][7]
                                z[1,32] = importances["importances_std"][8]
                                println(z[end, :])
                            else
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
                                im1 = importances["importances_mean"][1]
                                im2 = importances["importances_mean"][2]
                                im3 = importances["importances_mean"][3]
                                im4 = importances["importances_mean"][4]
                                im5 = importances["importances_mean"][5]
                                im6 = importances["importances_mean"][6]
                                im7 = importances["importances_mean"][7]
                                im8 = importances["importances_mean"][8]
                                sd1 = importances["importances_std"][1]
                                sd2 = importances["importances_std"][2]
                                sd3 = importances["importances_std"][3]
                                sd4 = importances["importances_std"][4]
                                sd5 = importances["importances_std"][5]
                                sd6 = importances["importances_std"][6]
                                sd7 = importances["importances_std"][7]
                                sd8 = importances["importances_std"][8]
                                z = vcat(z, [hls it alph act sol lr itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 im8 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8])
                                println(z[end, :])
                            end
                            println("End of ", itr, " iterations")
                            itr += 1
                        end
                    end
                end
            end
        end
    end
    z_df = DataFrame(layers = z[:,1], maxIt = z[:,2], alpha = z[:,3], act = z[:,4], solver = z[:,5], lr = z[:,6], f1_train = z[:,7], mcc_train = z[:,8], f1_val = z[:,9], mcc_val = z[:,10], f1_3Ftrain = z[:,11], f1_pest = z[:,12], mcc_pest = z[:,13], recall = z[:,14], state = z[:,15], model = z[:,16], im1 = z[:,17], im2 = z[:,18], im3 = z[:,19], im4 = z[:,20], im5 = z[:,21], im6 = z[:,22], im7 = z[:,23], im8 = z[:,24], sd1 = z[:,25], sd2 = z[:,26], sd3 = z[:,27], sd4 = z[:,28], sd5 = z[:,29], sd6 = z[:,30], sd7 = z[:,31], sd8 = z[:,32])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimMLP(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_MLP1_noFilter.csv"
CSV.write(savePath, optiSearch_df)

#===============================================================================#
