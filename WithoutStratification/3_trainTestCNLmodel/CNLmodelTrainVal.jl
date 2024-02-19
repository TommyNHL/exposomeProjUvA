VERSION
using Pkg
#Pkg.add("StatsPlots")
#Pkg.add("Plots")
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
#Pkg.add(PackageSpec(url=""))
using Random
using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots
#using PyPlot
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#using PyCall, Conda                 #using python packages
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using ScikitLearn.GridSearch: GridSearchCV

# inputing 4862 x (3+15994+1)
# columns: SMILES, INCHIKEY, CNLs, predictRi
#inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOnlyCocamides.csv", DataFrame)
inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOutsideCocamides.csv", DataFrame)

function partitionTrainVal(df, ratio = 0.7)
    noOfRow = nrow(df)
    idx = shuffle(1:noOfRow)
    train_idx = view(idx, 1:floor(Int, ratio*noOfRow))
    test_idx = view(idx, (floor(Int, ratio*noOfRow)+1):noOfRow)
    df[train_idx,:], df[test_idx,:]
end

# give 3+15994+1 = 15998 columns
trainSet, valSet = partitionTrainVal(inputDB, 0.7)  # 70% train 30% Val/Test
size(trainSet)
size(valSet)

# ouputing trainSet df 0.7 x (3+15994+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamidesTrain_withoutStratification.csv"
CSV.write(savePath, trainSet)

# ouputing trainSet df 0.3 x (3+15994+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamidesVal_withoutStratification.csv"
CSV.write(savePath, valSet)

# 15998 -> 15994 columns
## data for model training
x_train = deepcopy(trainSet)
select!(x_train, Not([:ENTRY, :SMILES, :INCHIKEY, :FPpredictRi]));
size(x_train)
y_train = deepcopy(trainSet[:, end])
size(y_train)

## data for internal model validation
x_val = deepcopy(valSet)
select!(x_val, Not([:ENTRY, :SMILES, :INCHIKEY, :FPpredictRi]));
size(x_val)
y_val = deepcopy(valSet[:, end])
size(y_val)

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

# modeling, 252 times
function optimRandomForestRegressor(Features, RIs)
    #leaf_r = [collect(4:2:10);15;20]
    leaf_r = [collect(8:8:32)]
    #tree_r = vcat(collect(50:50:400),collect(500:100:1000))
    tree_r = vcat(collect(50:50:300))
    z = zeros(1,6)
    itr = 1
    for l = 1:length(leaf_r)
        for t = 1:length(tree_r)
            for state = 1:3
                println("itr=", itr, ", leaf=", l, ", tree=", t, ", s=", state)
                leaf = leaf_r[l]
                tree = tree_r[t]
                MaxFeat = Int64(ceil(size(Features,2)/3))
                ## Regression ##
                Xx_train, Xx_test, Yy_train, Yy_test = train_test_split(Features, RIs, test_size=0.20, random_state=42);
                reg = RandomForestRegressor(n_estimators=tree, min_samples_leaf=leaf, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=42)
                fit!(reg, Xx_train, Yy_train)
                if itr == 1
                    z[1,1] = leaf
                    z[1,2] = tree
                    z[1,3] = state
                    z[1,4] = score(reg, Xx_train, Yy_train)
                    acc5_train = cross_val_score(reg, Xx_train, Yy_train; cv = 3)
                    z[1,5] = avgAcc(acc5_train, 3)
                    z[1,6] = score(reg, Xx_test, Yy_test)
                else
                    acc5_train = cross_val_score(reg, Xx_train, Yy_train; cv = 3)
                    z = vcat(z,[leaf tree state score(reg, Xx_train, Yy_train) avgAcc(acc5_train, 3) score(reg, Xx_test, Yy_test)])
                end
                #println("End of $itr iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], state=z[:,3], accuracy_train = z[:,4], avgAccuracy_train = z[:,5], accuracy_test = z[:,6])
    z_df_sorted = sort(z_df, [:avgAccuracy_train, :accuracy_test], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestRegressor(x_train, y_train)

#= model = RandomForestRegressor()
param_dist = Dict(
      "n_estimators" => 50:50:300, 
      #"max_depth" => 2:2:10, 
      "min_samples_leaf" => 8:8:32, 
      "max_features" => [Int64(ceil(size(x_train,2)/3))], 
      "n_jobs" => [-1], 
      "oob_score" => [true], 
      "random_state" => [1]
      )
gridsearch = GridSearchCV(model, param_dist)
@time fit!(gridsearch, Matrix(x_train), Vector(y_train))
println("Best parameters: $(gridsearch.best_params_)") =#

model = RandomForestRegressor(
      n_estimators = 300, 
      #max_depth = 10, 
      min_samples_leaf = 8, 
      max_features = Int64(ceil(size(x_train,2)/3)), 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42
      )
fit!(model, Matrix(x_train), Vector(y_train))

# saving model
modelSavePath = "D:\\1_model\\CocamideExtended_CNLsRi_RFwithoutStratification.joblib"
jl.dump(model, modelSavePath, compress = 5)


# training performace, CNL-predictedRi vs. FP-predictedRi
predictedRi_train = predict(model, Matrix(x_train))
trainSet[!, "CNLpredictRi"] = predictedRi_train
# save, ouputing trainSet df 0.7 x (3+15994+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamidesTrain_RFwithoutStratification_withCNLPredictedRi.csv"
CSV.write(savePath, trainSet)

maxAE_train, MSE_train, RMSE_train = errorDetermination(y_train, predictedRi_train)
rSquare_train = rSquareDetermination(y_train, predictedRi_train)
## accuracy
acc1_train = score(model, Matrix(x_train), Vector(y_train))
acc5_train = cross_val_score(model, Matrix(x_train), Vector(y_train); cv = 5)
avgAcc_train = avgAcc(acc5_train, 5)

# internal validation
predictedRi_val = predict(model, Matrix(x_val))
valSet[!, "CNLpredictRi"] = predictedRi_val
# save, ouputing trainSet df 0.7 x (3+15994+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamidesVal_RFwithoutStratification_withCNLPredictedRi.csv"
CSV.write(savePath, valSet)

maxAE_val, MSE_val, RMSE_val = errorDetermination(y_val, predictedRi_val)
rSquare_val = rSquareDetermination(y_val, predictedRi_val)
## accuracy
acc1_val = score(model, Matrix(x_val), Vector(y_val))
acc5_val = cross_val_score(model, Matrix(x_val), Vector(y_val); cv = 5)
avgAcc_val = avgAcc(acc5_val, 5)

# plots
plotTrain = marginalkde(
        y_train, 
        predictedRi_train, 
        xlabel = "FP-derived Ri values", 
        ylabel = "CNL-derived Ri values", 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
plot!(plotTrain.spmap[:contour], 
        y_train -> y_train, c=:red, 
        label = false, 
        title = "RF Model Training Without Stratification", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
        # Saving
savefig(plotTrain, "D:\\2_output\\CNLRiPrediction_RFTrainWithoutStratification.png")

plotVal = marginalkde(
        y_val, 
        predictedRi_val, 
        xlabel = "FP-derived Ri values", 
        ylabel = "CNL-derived Ri values", 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
plot!(plotVal.spmap[:contour], 
        y_val -> y_val, c=:red, 
        label = false, 
        title = "RF Model Val Without Stratification", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
        # Saving
savefig(plotVal, "D:\\2_output\\CNLRiPrediction_RFValWithoutStratification.png")