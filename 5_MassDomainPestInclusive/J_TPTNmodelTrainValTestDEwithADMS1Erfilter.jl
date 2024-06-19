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

# inputing 3391333 x 4+8+1+2+1+1+2+2 df
inputDB = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainYesDFwithhl_all.csv", DataFrame)
sort!(inputDB, [:ENTRY])
insertcols!(inputDB, 10, ("MatchRatio"=>float(0)))
inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
describe(inputDB[inputDB.LABEL .== 1, :])
inputDB = inputDB[inputDB.MS1Error .>= float(-0.001), :]
inputDB = inputDB[inputDB.MS1Error .<= float(0.001), :]
for i = 1:size(inputDB, 1)
    inputDB[i, "RefMatchFragRatio"] = log10(inputDB[i, "RefMatchFragRatio"])
    inputDB[i, "UsrMatchFragRatio"] = log10(inputDB[i, "UsrMatchFragRatio"])
    inputDB[i, "FinalScoreRatio"] = log10(inputDB[i, "FinalScoreRatio"])
    inputDB[i, "MatchRatio"] = inputDB[i, "DirectMatch"] - inputDB[i, "ReversMatch"]
end
# save, ouputing 485631 x 21+1 df, 0:334321; 1:151310 = 0.7263; 1.6048
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl0d5FinalScoreRatio2DE2Filter_all.csv"
CSV.write(savePath, inputDB)
inputDB[inputDB.LABEL .== 1, :]

describe((inputDB_test))[1:14, :]
# inputing 847838 x 4+8+1+2+1+1+2+2 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestYesDFwithhl_all.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
insertcols!(inputDB_test, 10, ("MatchRatio"=>float(0)))
inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
describe(inputDB_test[inputDB_test.LABEL .== 1, :])
inputDB_test = inputDB_test[inputDB_test.MS1Error .>= float(-0.001), :]
inputDB_test = inputDB_test[inputDB_test.MS1Error .<= float(0.001), :]
for i = 1:size(inputDB_test, 1)
    inputDB_test[i, "RefMatchFragRatio"] = log10(inputDB_test[i, "RefMatchFragRatio"])
    inputDB_test[i, "UsrMatchFragRatio"] = log10(inputDB_test[i, "UsrMatchFragRatio"])
    inputDB_test[i, "FinalScoreRatio"] = log10(inputDB_test[i, "FinalScoreRatio"])
    inputDB_test[i, "MatchRatio"] = inputDB_test[i, "DirectMatch"] - inputDB_test[i, "ReversMatch"]
end
# save, ouputing 121946 x 21+1 df, 0:83981; 1:37965 = 0.7260; 1.6060
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDFwithhl0d5FinalScoreRatio2DE2Filter_all.csv"
CSV.write(savePath, inputDB_test)
inputDB_test[inputDB_test.LABEL .== 1, :]

# 607577 x 22 df; 
# 485631+121946= 607577, 0:418302; 1:189275 = 
inputDBInputDB_test = vcat(inputDB, inputDB_test)
sort!(inputDBInputDB_test, [:ENTRY])
inputDBInputDB_test[inputDBInputDB_test.LABEL .== 1, :]


#TP/TN prediction
inputDB_pest2 = CSV.read("F:\\UvA\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
sort!(inputDB_pest2, [:ENTRY])
insertcols!(inputDB_pest2, 10, ("MatchDiff"=>float(0)))
inputDB_pest2 = inputDB_pest2[inputDB_pest2.FinalScoreRatio .>= float(0.5), :]
inputDB_pest2 = inputDB_pest2[inputDB_pest2.Leverage .<= 0.14604417882015916, :]
describe(inputDB_pest2[inputDB_pest2.LABEL .== 0, :])
describe(inputDB_pest2[inputDB_pest2.LABEL .== 1, :])
inputDB_pest2 = inputDB_pest2[inputDB_pest2.MS1Error .>= float(-0.061), :]
inputDB_pest2 = inputDB_pest2[inputDB_pest2.MS1Error .<= float(0.058), :]
for i = 1:size(inputDB_pest2, 1)
    inputDB_pest2[i, "RefMatchFragRatio"] = log10(inputDB_pest2[i, "RefMatchFragRatio"])
    inputDB_pest2[i, "UsrMatchFragRatio"] = log10(inputDB_pest2[i, "UsrMatchFragRatio"])
    inputDB_pest2[i, "FinalScoreRatio"] = log10(inputDB_pest2[i, "FinalScoreRatio"])
    inputDB_pest2[i, "MatchDiff"] = inputDB_pest2[i, "DirectMatch"] - inputDB_pest2[i, "ReversMatch"]
end
# save, ouputing 4757 x 18+1 df, 0:3423; 1:1334 = 0.6949; 1.7830
savePath = "F:\\UvA\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_dataframeTPTNModeling_testDFwithhl0d5FinalScoreRatioDEFilter.csv"
CSV.write(savePath, inputDB_pest2)
inputDB_pest2[inputDB_pest2.LABEL .== 1, :]

Yy_train = deepcopy(inputDB[:, end-4])  # 0.7263; 1.6048
sampleW = []
for w in Vector(Yy_train)
    if w == 0
        push!(sampleW, 0.7263)
    elseif w == 1
        push!(sampleW, 1.6048)
    end
end 

Yy_val = deepcopy(inputDB_test[:, end-4])  # 0.7260; 1.6060
sampletestW = []
for w in Vector(Yy_val)
    if w == 0
        push!(sampletestW, 0.7260)
    elseif w == 1
        push!(sampletestW, 1.6060)
    end
end 


Yy_test2 = deepcopy(inputDB_pest2[:, end-1])  # 0.7183; 1.6453
samplepest2W = []
for w in Vector(Yy_test2)
    if w == 0
        push!(samplepest2W, 0.6949)
    elseif w == 1
        push!(samplepest2W, 1.7830)
    end
end 

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
describe((inputDB))[vcat(5,6,8,9,10, 13, end-5), :]
describe((inputDB_test))[vcat(5,6,8,9,10, 13, end-5), :]
describe((inputDB_pest2))[vcat(5,6,8,9,10, 13, end-2), :]

function optimRandomForestClass(inputDB, inputDB_test, inputDB_pest2)
    leaf_r = vcat(collect(2:2:40))
    tree_r = vcat(collect(50:50:400))
    depth_r = vcat(collect(30:10:100))
    split_r = vcat(collect(2:1:10))
    model_r = vcat(9, 8)
    rs = 42
    z = zeros(1,14)
    itr = 1
    while itr < 129
        l = rand(leaf_r)
        t = rand(tree_r)
        d = rand(depth_r)
        r = rand(split_r)
        for mod in model_r
            println("itr=", itr, ", leaf=", l, ", tree=", t, ", depth=", d, ", minSsplit=", r)
            println("## loading in data ##")
            M_train = inputDB
            M_val = inputDB_test
            M_pest2 = inputDB_pest2
            if mod == 8
                Xx_train = deepcopy(M_train[:, vcat(5,6,8,9,10, 13)])
                Xx_val = deepcopy(M_val[:, vcat(5,6,8,9,10, 13)])
                Xx_test2 = deepcopy(M_pest2[:, vcat(5,6,8,9,10, 13)])
            elseif mod == 9
                Xx_train = deepcopy(M_train[:, vcat(5,6,8,9,10, 13, end-5)])
                Xx_val = deepcopy(M_val[:, vcat(5,6,8,9,10, 13, end-5)])
                Xx_test2 = deepcopy(M_pest2[:, vcat(5,6,8,9,10, 13, end-2)])
            end
            Yy_train = deepcopy(M_train[:, end-4])
            Yy_val = deepcopy(M_val[:, end-4])
            Yy_test2 = deepcopy(M_pest2[:, end-1])
            println("## Classification ##")
            reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, n_jobs=-1, oob_score =true, random_state=rs, class_weight=Dict(0=>0.7263, 1=>1.6048))  # 0.7263; 1.6048
            println("## fit ##")
            fit!(reg, Matrix(Xx_train), Vector(Yy_train))
            if itr == 1
                z[1,1] = l
                z[1,2] = t
                z[1,3] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                z[1,4] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                z[1,5] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                z[1,6] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                z[1,7] = avgScore(f1_10_train, 3)
                z[1,8] = score(reg, Matrix(Xx_test2), Vector(Yy_test2))
                z[1,9] = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
                z[1,10] = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
                z[1,11] = rs
                z[1,12] = d
                z[1,13] = r
                z[1,14] = mod
                println(z[end, :])
            else
                itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                traincvtrain = avgScore(f1_10_train, 3) 
                itest = score(reg, Matrix(Xx_test2), Vector(Yy_test2))
                f1s = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
                mccs = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
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

optiSearch_df = optimRandomForestClass(inputDB, inputDB_test, inputDB_pest2)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatioDE3_RFwithhlnew2Compare45_all.csv"
CSV.write(savePath, optiSearch_df)

function optimLR(inputDB, inputDB_test, inputDB_pest2)
    penalty_r = vcat("l1", "l2")
    solver_rs = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    solver_r = vcat(collect(1:1:5))
    c_values_r = vcat(1000, 500, 100, 50, 10, 1.0)
    model_r = vcat(9, 8)
    rs = 42
    z = zeros(1,13)
    itr = 1
    pn = float(0)
    while itr < 65
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
            M_pest2 = inputDB_pest2
            if mod == 8
                Xx_train = deepcopy(M_train[:, vcat(5,6,8,9,10, 13)])
                Xx_val = deepcopy(M_val[:, vcat(5,6,8,9,10, 13)])
                Xx_test2 = deepcopy(M_pest2[:, vcat(5,6,8,9,10, 13)])
            elseif mod == 9
                Xx_train = deepcopy(M_train[:, vcat(5,6,8,9,10, 13, end-5)])
                Xx_val = deepcopy(M_val[:, vcat(5,6,8,9,10, 13, end-5)])
                Xx_test2 = deepcopy(M_pest2[:, vcat(5,6,8,9,10, 13, end-2)])
            end
            Yy_train = deepcopy(M_train[:, end-4])
            Yy_val = deepcopy(M_val[:, end-4])
            Yy_test2 = deepcopy(M_pest2[:, end-1])
            println("## Classification ##")
            reg = LogisticRegression(penalty=p, C=c, solver=solver_rs[s], max_iter=5000, random_state=rs, class_weight=Dict(0=>0.7263, 1=>1.6048))  # 0.7263; 1.6048
            println("## fit ##")
            fit!(reg, Matrix(Xx_train), Vector(Yy_train))
            if itr == 1
                z[1,1] = pn
                z[1,2] = c
                z[1,3] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                z[1,4] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                z[1,5] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                z[1,6] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                z[1,7] = avgScore(f1_10_train, 3)
                z[1,8] = score(reg, Matrix(Xx_test2), Vector(Yy_test2))
                z[1,9] = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
                z[1,10] = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepes2W)
                z[1,11] = rs
                z[1,12] = s
                z[1,13] = mod
                println(z[end, :])
            else
                itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                traincvtrain = avgScore(f1_10_train, 3) 
                itest = score(reg, Matrix(Xx_test2), Vector(Yy_test2))
                f1s = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
                mccs = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
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

optiSearch_df = optimLR(inputDB, inputDB_test, inputDB_pest2)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatioDE3_LRwithhlnew2Compare4_all.csv"
CSV.write(savePath, optiSearch_df)


function optimGradientBoostClass(inputDB, inputDB_test, inputDB_pest2)
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
    model_r = vcat(9, 8)
    rs = 42
    z = zeros(1,15)
    itr = 1
    while itr < 41
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
            M_pest2 = inputDB_pest2
            if mod == 8
                Xx_train = deepcopy(M_train[:, vcat(5,6,8,9,10, 13)])
                Xx_val = deepcopy(M_val[:, vcat(5,6,8,9,10, 13)])
                Xx_test2 = deepcopy(M_pest2[:, vcat(5,6,8,9,10, 13)])
            elseif mod == 9
                Xx_train = deepcopy(M_train[:, vcat(5,6,8,9,10, 13, end-5)])
                Xx_val = deepcopy(M_val[:, vcat(5,6,8,9,10, 13, end-5)])
                Xx_test2 = deepcopy(M_pest2[:, vcat(5,6,8,9,10, 13, end-2)])
            end
            Yy_train = deepcopy(M_train[:, end-4])
            Yy_val = deepcopy(M_val[:, end-4])
            Yy_test2 = deepcopy(M_pest2[:, end-1])
            println("## Classification ##")
            reg = GradientBoostingClassifier(learning_rate=lr, n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=r, random_state=rs)
            println("## fit ##")
            fit!(reg, Matrix(Xx_train), Vector(Yy_train), sample_weight=sampleW)
            if itr == 1
                z[1,1] = lr
                z[1,2] = l
                z[1,3] = t
                z[1,4] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                z[1,5] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                z[1,6] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                z[1,7] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                z[1,8] = avgScore(f1_10_train, 3)
                z[1,9] = score(reg, Matrix(Xx_test2), Vector(Yy_test2))
                z[1,10] = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
                z[1,11] = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
                z[1,12] = rs
                z[1,13] = d
                z[1,14] = r
                z[1,15] = mod
                println(z[end, :])
            else
                itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleW)
                ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                println("## CV ##")
                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                traincvtrain = avgScore(f1_10_train, 3) 
                itest = score(reg, Matrix(Xx_test2), Vector(Yy_test2))
                f1s = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
                mccs = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplepest2W)
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

optiSearch_df = optimGradientBoostClass(inputDB, inputDB_test, inputDB_pest2)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\hyperparameterTuning_TPTNwithAbsDeltaRi3F_0d5FinalScoreRatioDE3_GBMwithhlnew2Compare1_all.csv"
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
