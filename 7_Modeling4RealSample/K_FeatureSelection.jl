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

# inputing 1686319 x 22 df
# 0: 1535009; 1: 151310 = 0.5493; 5.5724
trainDEFSDf = CSV.read("F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
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
testDEFSDf = CSV.read("F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
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
noTeaDEFSDf = CSV.read("F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
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
TeaDEFSDf = CSV.read("F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
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
describe((trainDEFSDf))[vcat(5,6,7,9,10,13,14, 17), :]
describe((testDEFSDf))[vcat(5,6,9,10,13,14, end-5), :]
describe((noTeaDEFSDf))[vcat(5,6,9,10,13,14, end-2), :]
describe((TeaDEFSDf))[vcat(5,6,9,10,13,14, end-2), :]
describe((TeaDEFSDf))
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    leaf_r = vcat(collect(2:20:102))  # 5
    depth_r = vcat(collect(2:10:62))  # 7
    split_r = vcat(collect(2:20:102))  # 5
    rs = 42
    z = zeros(1,22)
    itr = 1
    t = 50
    mod = 0
    rank = vcat(5,6,7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    for l in leaf_r
        for d in depth_r
            for r in split_r
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
                    z[1,22] = importances["importances_mean"][8]
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
                    z = vcat(z, [l t d r itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 im8])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], depth = z[:,3], minSplit = z[:,4], f1_train = z[:,5], mcc_train = z[:,6], f1_val = z[:,7], mcc_val = z[:,8], f1_3Ftrain = z[:,9], f1_pest = z[:,10], mcc_pest = z[:,11], recall = z[:,12], state = z[:,13], model = z[:,14], im1 = z[:,15], im2 = z[:,16], im3 = z[:,17], im4 = z[:,18], im5 = z[:,19], im6 = z[:,20], im7 = z[:,21], im8 = z[:,22])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestClass(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_RF1_noFilter.csv"
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
