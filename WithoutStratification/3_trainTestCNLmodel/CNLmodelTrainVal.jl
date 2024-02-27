VERSION
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add("Plots")
#Pkg.add("ProgressBars")
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
#Pkg.add(PackageSpec(url=""))
using Random
#using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots
#using ProgressBars
#using PyPlot
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#using PyCall, Conda                 #using python packages
#pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

# inputing 693685*0.3 x 1+1+1+15961+1 df = 208106 x 15965
# columns: ENTRY, INCHIKEY, ISOTOPICMASS, CNLs, predictRi
inputDB_test = CSV.read("F:\\dataframe_dfTestSetWithStratification.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
# inputing 693685*0.7 x 1+1+1+15961+1 df = 485579 x 15965
inputDB = CSV.read("F:\\dataframe_dfTrainSetWithStratification.csv", DataFrame)
sort!(inputDB, [:ENTRY])

# internal train/test split
X = deepcopy(inputDB[:, 3:end-1])
size(X)
Y = deepcopy(inputDB[:, end])
size(Y)

function partitionTrainVal(df, ratio = 0.67)
    noOfRow = nrow(df)
    idx = shuffle(1:noOfRow)
    train_idx = view(idx, 1:floor(Int, ratio*noOfRow))
    test_idx = view(idx, (floor(Int, ratio*noOfRow)+1):noOfRow)
    df[train_idx,:], df[test_idx,:]
end

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

# modeling, 6 x 6 = 36 times
function optimRandomForestRegressor(df_train)
    #leaf_r = [collect(4:2:10);15;20]
    leaf_r = vcat(collect(4:2:8), collect(12:4:20))
    #tree_r = vcat(collect(50:50:400),collect(500:100:1000))
    tree_r = collect(50:50:300)
    z = zeros(1,6)
    itr = 1
    while itr < 6
        l = rand(leaf_r)
        t = rand(tree_r)
        println("itr=", itr, ", leaf=", l, ", tree=", t)
        MaxFeat = Int64(ceil((size(df_train,2)-1)/3))
        println("## split ##")
        M_train, M_val = partitionTrainVal(df_train, 0.67)
        Xx_train = deepcopy(M_train[:, 3:end-1])
        Yy_train = deepcopy(M_train[:, end])
        Xx_val = deepcopy(M_val[:, 3:end-1])
        Yy_val = deepcopy(M_val[:, end])
        println("## Regression ##")
        reg = RandomForestRegressor(n_estimators=t, min_samples_leaf=l, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=42)
        println("## fit ##")
        fit!(reg, Matrix(Xx_train), Vector(Yy_train))
        if itr == 1
            z[1,1] = l
            z[1,2] = t
            z[1,3] = score(reg, Matrix(Xx_train), Vector(Yy_train))
            z[1,4] = score(reg, Matrix(df_train[:, 3:end-1]), Vector(df_train[:, end]))
            println("## CV ##")
            acc5_train = cross_val_score(reg, Matrix(df_train[:, 3:end-1]), Vector(df_train[:, end]); cv = 3)
            z[1,5] = avgAcc(acc5_train, 3)
            z[1,6] = score(reg, Matrix(Xx_val), Vector(Yy_val))
            println(z[end, :])
        else
            println("## CV ##")
            itrain= score(reg, Matrix(Xx_train), Vector(Yy_train)) 
            traintrain = score(reg, Matrix(df_train[:, 3:end-1]), Vector(df_train[:, end]))
            acc5_train = cross_val_score(reg, Matrix(df_train[:, 3:end-1]), Vector(df_train[:, end]); cv = 3)
            traincvtrain = avgAcc(acc5_train, 3) 
            ival = score(reg, Matrix(Xx_val), Vector(Yy_val)) 
            z = vcat(z, [l t itrain traintrain traincvtrain ival])
            println(z[end, :])
        end
        println("End of ", itr, " iterations")
        itr += 1
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], accuracy_3Ftrain = z[:,3], accuracy_train = z[:,4], avgAccuracy3FCV_train = z[:,5], accuracy_val = z[:,6])
    z_df_sorted = sort(z_df, [:accuracy_val, :avgAccuracy3FCV_train, :accuracy_train, :accuracy_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestRegressor(inputDB)

# save, ouputing 180 x 8 df
savePath = "F:\\hyperparameterTuning_RFwithStratification.csv"
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

model = RandomForestRegressor(
      n_estimators = 250, 
      #max_depth = 10, 
      min_samples_leaf = 4, 
      max_features = Int64(ceil((size(inputDB[:, 3:end-1],2)-1)/3)), 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42
      )
fit!(model, Matrix(inputDB[:, 3:end-1]), Vector(inputDB[:, end]))

# saving model
modelSavePath = "F:\\CocamideExtended_CNLsRi_RFwithStratification.joblib"
jl.dump(model, modelSavePath, compress = 5)

# training performace, CNL-predictedRi vs. FP-predictedRi
predictedRi_train = predict(model, Matrix(inputDB[:, 3:end-1]))
inputDB[!, "CNLpredictRi"] = predictedRi_train
# save, ouputing trainSet df 0.7 x (3+15994+1)
savePath = "F:\\dataframe_dfTrainSetWithStratification_withCNLPredictedRi.csv"
CSV.write(savePath, inputDB)

maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB[:, end-1], predictedRi_train)
rSquare_train = rSquareDetermination(inputDB[:, end-1], predictedRi_train)
## accuracy
acc1_train = score(model, Matrix(inputDB[:, 3:end-2]), Vector(inputDB[:, end-1]))
acc5_train = cross_val_score(model, Matrix(inputDB[:, 3:end-2]), Vector(inputDB[:, end-1]); cv = 3)
avgAcc_train = avgAcc(acc5_train, 3)

# model validation
#load a model
# requires python 3.11 or 3.12
modelRF_CNL = jl.load("F:\\CocamideExtended_CNLsRi_RFwithStratification.joblib")
model = jl.load("F:\\CocamideExtended_CNLsRi_RFwithStratification.joblib")
size(modelRF_CNL)

predictedRi_test = predict(modelRF_CNL, Matrix(inputDB_test[:, 3:end-1]))
inputDB_test[!, "CNLpredictRi"] = predictedRi_test
# save, ouputing testSet df 0.3 x (3+15994+1)
savePath = "F:\\dataframe_dfTestSetWithStratification_withCNLPredictedRi.csv"
CSV.write(savePath, inputDB_test)

maxAE_val, MSE_val, RMSE_val = errorDetermination(inputDB_test[:, end-1], predictedRi_test)
rSquare_val = rSquareDetermination(inputDB_test[:, end-1], predictedRi_test)
## accuracy
acc1_val = score(modelRF_CNL, Matrix(inputDB_test[:, 3:end-2]), Vector(inputDB_test[:, end-1]))
acc5_val = cross_val_score(modelRF_CNL, Matrix(inputDB_test[:, 3:end-2]), Vector(inputDB_test[:, end-1]); cv = 3)
avgAcc_val = avgAcc(acc5_val, 3)

# plots
# inputing dfs for separation of the cocamides and non-cocamides datasets
## 5364 x 931 df 
inputCocamidesTrain = CSV.read("F:\\CocamideExtWithStartification_Fingerprints_train.csv", DataFrame)
sort!(inputCocamidesTrain, :SMILES)

## 947 x 931 df
inputCocamidesTest = CSV.read("F:\\CocamideExtWithStratification_Fingerprints_test.csv", DataFrame)
sort!(inputCocamidesTest, :SMILES)

# comparing, 30684 x 793 df
inputAllFPDB = CSV.read("F:\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
sort!(inputAllFPDB, [:INCHIKEY, :SMILES])

function id2id(plotdf, i)
    inchikeyID = plotdf[i, "INCHIKEY"]
    idx = findall(inputAllFPDB.INCHIKEY .== inchikeyID)
    return inputAllFPDB[idx[end:end], "SMILES"][1]
end

function cocamidesOrNot(plotdf, i)
    if (id2id(plotdf, i) in Array(inputCocamidesTrain[:, "SMILES"]) || id2id(plotdf, i) in Array(inputCocamidesTest[:, "SMILES"]))
        return true
    else
        return false
    end
end

function findDots(plotdf, i)
    if (cocamidesOrNot(plotdf, i) == true)
        return plotdf[i, "predictRi"], plotdf[i, "CNLpredictRi"]
    end
end

trainCocamide = []
trainNonCocamide = []
inputDB[!, "Cocamides"] .= ""
for i in 1:size(inputDB, 1)
    if (cocamidesOrNot(inputDB, i) == true)
        inputDB[i, "Cocamides"] = "yes"
        push!(trainCocamide, i)
    elseif (cocamidesOrNot(inputDB, i) == false)
        inputDB[i, "Cocamides"] = "no"
        push!(trainNonCocamide, i)
    end
end
savePath = "F:\\dataframe_dfTrainSetWithStratification_withCNLPredictedRi_withCocamides.csv"
CSV.write(savePath, inputDB)

testCocamide = []
testNonCocamide = []
inputDB_test[!, "Cocamides"] .= ""
for i in 1:size(inputDB_test, 1)
    if (cocamidesOrNot(inputDB_test, i) == true)
        inputDB_test[i, "Cocamides"] = "yes"
        push!(testCocamide, i)
    elseif (cocamidesOrNot(inputDB_test, i) == false)
        inputDB_test[i, "Cocamides"] = "no"
        push!(testNonCocamide, i)
    end
end
savePath = "F:\\dataframe_dfTestSetWithStratification_withCNLPredictedRi_withCocamides.csv"
CSV.write(savePath, inputDB_test)

plotTrain = marginalkde(
        inputDB[:, end-1], 
        predictedRi_train, 
        xlabel = "FP-derived Ri values", 
        ylabel = "CNL-derived Ri values", 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
plot!(plotTrain.spmap[:contour], 
        Array(inputDB[:, end-1]) -> Array(inputDB[:, end-1]), c=:red, 
        label = false, 
        title = "RF Model Training With Stratification", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
scatter!(inputDB[trainCocamide, end-1], predictedRi_train[trainCocamide], 
        markershape = :star, 
        c = :yellow, 
        label = "Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
scatter!(inputDB[trainNonCocamide, end-1], predictedRi_train[trainNonCocamide], 
        markershape = :star, 
        c = :orange, 
        label = "Non-Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
        # Saving
savefig(plotTrain, "F:\\CNLRiPrediction_RFTrainWithStratification.png")

plotTest = marginalkde(
        inputDB_test[:, end-1], 
        predictedRi_test, 
        xlabel = "FP-derived Ri values", 
        ylabel = "CNL-derived Ri values", 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
plot!(plotTest.spmap[:contour], 
        Array(inputDB_test[:, end-1]) -> Array(inputDB_test[:, end-1]), c=:red, 
        label = false, 
        title = "RF Model Test With Stratification", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
scatter!(inputDB_test[testCocamide, end-1], predictedRi_test[testCocamide], 
        markershape = :star, 
        c = :yellow, 
        label = "Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
scatter!(inputDB_test[testNonCocamide, end-1], predictedRi_test[testNonCocamide], 
        markershape = :star, 
        c = :orange, 
        label = "Non-Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
        # Saving
savefig(plotTest, "F:\\CNLRiPrediction_RFTestWithStratification.png")