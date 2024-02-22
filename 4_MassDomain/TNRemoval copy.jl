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
using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots
using ProgressBars
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

# inputing 693677 x 3+21567 df
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputTPTNdf = CSV.read("D:\\Cand_search_rr0_0612_TEST_100-400_extractedWithDeltaRi.csv", DataFrame)
sort!(inputTPTNdf, [:LABEL, :INCHIKEY_ID, :DeltaRi, :FinalScore])

inputDfOnlyTP = inputTPTNdf[inputTPTNdf.LABEL .== 1, :]
inputDfOnlyTN = inputTPTNdf[inputTPTNdf.LABEL .== 0, :]

# Train/Test Split by Leverage
## calculating how large the portion of TN is needed to be removed
X = deepcopy(inputDfOnlyTN[:, 2:end-1])  # 693677 x 790 df
size(X)
Y = deepcopy(inputDfOnlyTN[:, end])  #693677,
size(Y)

using ProgressBars
using LinearAlgebra
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split
function leverage_dist(X)   # Set x1 and x2 to your FPs variables
    h = zero(xxx,1)
    for i in ProgressBar(1: size(X,1)) #check dimensions
        x = X[i,:] 
        hi = x'*pinv(X'*X)*x
        #push!(h,hi)
        h[i,1] = hi
    end
    return h
end

h = leverage_dist(Matrix(X))

function strat_split(leverage=h; limits = limits)
    n = length(leverage)
    bin = collect(1:n)
    for i = 1: (length(limits)-1)
        bin[limits[i].<= leverage].= i
    end
    X_train, X_test, y_train, y_test = train_test_split(collect(1:length(leverage)), leverage, test_size = 0.30, random_state = 42, stratify = bin)
    return  X_train, X_test, y_train, y_test
end

X_trainIdx, X_testIdx, train_lev, test_lev = strat_split(h, limits = collect(0.0:0.2:1))
inputTPTNdf[!, "GROUP"] .= ""
inputTPTNdf[!, "Leverage"] .= float(0)
inputTPTNdf[X_trainIdx, "GROUP"] .= "train"
inputTPTNdf[X_testIdx, "GROUP"] .= "test"
count = 1
for i in X_trainIdx
  inputTPTNdf[i, "Leverage"] = train_lev[count]
    count += 1
end
count = 1
for i in X_testIdx
  inputTPTNdf[i, "Leverage"] = test_lev[count]
    count += 1
end

# output csv is a 693677 x 3+790+2 df
savePath = "D:\\Cand_search_rr0_0612_TEST_100-400_extractedWithDeltaRi_SplitByLeverage.csv"
CSV.write(savePath, inputTPTNdf)

function create_train_test_split_strat(total_df, y_data, X_trainIdx, X_testIdx, RiCol = true)
    #X_train_ind, X_test_ind, train_lev, test_lev = strat_split(leverage, limits = limits)
    # Create train test split of total DataFrame and dependent variables using the chosen parameters
    X_trainTPTN = total_df[X_trainIdx,:]
    X_testTPTN = total_df[X_testIdx,:]
    if (RiCol == true)
        Y_trainLabel = y_data[X_trainIdx]
        Y_testLabel = y_data[X_testIdx]
        return  X_trainTPTN, X_testTPTN, Y_trainLabel, Y_testLabel
    end
    # # Select train and test set of independent variables 
    # X_train = total_train[:, start_col_X_data:end]
    # X_test = total_test[:, start_col_X_data:end]
    return  X_trainTPTN, X_testTPTN
end

X_trainTPTN, X_testTPTN, Y_trainLabel, Y_testLabel = create_train_test_split_strat(X, Y, X_trainIdx, X_testIdx, true)

inputTPTNdf

df_info = inputTPTNdf[:, 1:1]
df_info
X_trainInfo, X_testInfo = create_train_test_split_strat(df_info, df_info, X_trainIdx, X_testIdx, false)

dfTrainSet = hcat(X_trainInfo, X_trainTPTN, Y_trainLabel)
dfTrainSet
# output csv is a 693677*0.7 x 3+22357+1 df
savePath = "D:\\dataframe_TPTNdfTrainSet.csv"
CSV.write(savePath, dfTrainSet)

dfTestSet = hcat(X_testInfo, X_testTPTN, Y_testLabel)
dfTestSet
# output csv is a 693677*0.3 x 3+22357+1 df
savePath = "D:\\dataframe_TPTNdfTestSet.csv"
CSV.write(savePath, dfTestSet)

# inputing 693677*0.3 x (3+22357+1)
# columns: SMILES, INCHIKEY, CNLs, predictRi
X_inputDB_test = deepcopy(dfTestSet[:, 2:end-1])
size(X_inputDB_test)
Y_inputDB_test = deepcopy(dfTestSet[:, end])
size(YY_inputDB_test)

# inputing 693677*0.7 x (3+22357+1)
X_inputDB_train = deepcopy(dfTrainSet[:, 2:end-1])
size(X_inputDB_train)
Y_inputDB_train = deepcopy(dfTrainSet[:, end])
size(Y_inputDB_train)

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

# modeling, 4 x 6 x 3 times
function optimRandomForestRegressor(df_train, df_test)
    #leaf_r = [collect(4:2:10);15;20]
    leaf_r = collect(8:8:32)
    #tree_r = vcat(collect(50:50:400),collect(500:100:1000))
    tree_r = collect(50:50:300)
    z = zeros(1,8)
    itr = 1
    for l in leaf_r
        for t in tree_r
            for state = 1:3
                println("itr=", itr, ", leaf=", l, ", tree=", t, ", s=", state)
                MaxFeat = Int64((ceil(size(df_train,2)-1)/3))
                println("## split ##")
                M_train, M_val = partitionTrainVal(df_train, 0.67)
                Xx_train = deepcopy(M_train[:, 4:end-1])
                Yy_train = deepcopy(M_train[:, end])
                Xx_val = deepcopy(M_val[:, 4:end-1])
                Yy_val = deepcopy(M_val[:, end])
                xx_test = deepcopy(df_test[:, 4:end-1])
                yy_test = deepcopy(df_test[:, end])
                println("## Regression ##")
                reg = RandomForestRegressor(n_estimators=t, min_samples_leaf=l, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=42)
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                if itr == 1
                    z[1,1] = l
                    z[1,2] = t
                    z[1,3] = state
                    z[1,4] = score(reg, Matrix(Xx_train), Vector(Yy_train))
                    z[1,5] = score(reg, Matrix(df_train[:, 4:end-1]), Vector(df_train[:, end]))
                    println("## CV ##")
                    acc5_train = cross_val_score(reg, Matrix(df_train[:, 4:end-1]), Vector(df_train[:, end]); cv = 3)
                    z[1,5] = avgAcc(acc5_train, 3)
                    z[1,6] = score(reg, Matrix(Xx_val), Vector(Yy_val))
                    z[1,7] = score(reg, Matrix(xx_test), Vector(yy_test))
                    println(z)
                else
                    println("## CV ##")
                    itrain= score(reg, Matrix(Xx_train), Vector(Yy_train)) 
                    traintrain = score(reg, Matrix(df_train[:, 4:end-1]), Vector(df_train[:, end]))
                    acc5_train = cross_val_score(reg, Matrix(df_train[:, 4:end-1]), Vector(df_train[:, end]); cv = 3)
                    traincvtrain = avgAcc(acc5_train, 3) 
                    ival = score(reg, Matrix(Xx_val), Vector(Yy_val)) 
                    etest = score(reg, Matrix(xx_test), Vector(yy_test))
                    z = vcat(z, [l t state itrain traintrain traincvtrain ival etest])
                    println(z)
                end
                #println("End of $itr iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], state=z[:,3], accuracy_3Ftrain = z[:,4], accuracy_train = z[:,5], avgAccuracy3FCV_train = z[:,6], accuracy_val = z[:,7],  accuracy_ext_test = z[:,8])
    z_df_sorted = sort(z_df, [:accuracy_ext_test, :accuracy_val, :avgAccuracy3FCV_train, :accuracy_train, :accuracy_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestRegressor(inputDB, inputDB_test)

# save, ouputing 72 x 8 df
savePath = "D:\\0_data\\hyperparameterTuning_RFwithStratification.csv"
CSV.write(savePath, optiSearch_df)

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
fit!(model, Matrix(inputDB[:, 4:end-1]), Vector(inputDB[:, end]))

# saving model
modelSavePath = "D:\\1_model\\CocamideExtended_CNLsRi_RFwithStratification.joblib"
jl.dump(model, modelSavePath, compress = 5)

# training performace, CNL-predictedRi vs. FP-predictedRi
predictedRi_train = predict(model, Matrix(inputDB[:, 4:end-1]))
inputDB[!, "CNLpredictRi"] = predictedRi_train
# save, ouputing trainSet df 0.7 x (3+15994+1)
savePath = "D:\\0_data\\dataframe_dfTrainSetWithStratification_withCNLPredictedRi.csv"
CSV.write(savePath, inputDB)

maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB[:, end], predictedRi_train)
rSquare_train = rSquareDetermination(inputDB[:, end], predictedRi_train)
## accuracy
acc1_train = score(model, Matrix(inputDB[:, 4:end-1]), Vector(inputDB[:, end]))
acc5_train = cross_val_score(model, Matrix(inputDB[:, 4:end-1]), Vector(inputDB[:, end]); cv = 5)
avgAcc_train = avgAcc(acc5_train, 5)

# model validation
#load a model
# requires python 3.11 or 3.12
modelRF_CNL = jl.load("D:\\1_model\\CocamideExtended_CNLsRi_RFwithStratification.joblib")
size(modelRF_CNL)

predictedRi_test = predict(modelRF_CNL, Matrix(inputDB_test[:, 4:end-1]))
inputDB_test[!, "CNLpredictRi"] = predictedRi_test
# save, ouputing trainSet df 0.7 x (3+15994+1)
savePath = "D:\\0_data\\dataframe_dfTestSetWithStratification_withCNLPredictedRi.csv"
CSV.write(savePath, inputDB_test)

maxAE_val, MSE_val, RMSE_val = errorDetermination(inputDB_test[:, end], predictedRi_test)
rSquare_val = rSquareDetermination(inputDB_test[:, end], predictedRi_test)
## accuracy
acc1_val = score(modelRF_CNL, Matrix(inputDB_test[:, 4:end-1]), Vector(inputDB_test[:, end]))
acc5_val = cross_val_score(modelRF_CNL, Matrix(inputDB_test[:, 4:end-1]), Vector(inputDB_test[:, end]); cv = 5)
avgAcc_val = avgAcc(acc5_val, 5)

# plots
# inputing dfs for separation of the cocamides and non-cocamides datasets
## 5364 x 931 df 
inputCocamidesTrain = CSV.read("D:\\0_data\\CocamideExtWithStartification_Fingerprints_train.csv", DataFrame)
sort!(inputCocamidesTrain, :SMILES)

## 947 x 931 df
inputCocamidesTest = CSV.read("D:\\0_data\\CocamideExtWithStartification_Fingerprints_test.csv", DataFrame)
sort!(inputCocamidesTest, :SMILES)

function cocamidesOrNot(plotdf, i)
    if (plotdf[i, "SMILES"] in Array(inputCocamidesTrain[:, "SMILES"]) || plotdf[i, "SMILES"] in Array(inputCocamidesTest[:, "SMILES"]))
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
inputDB[!, "Cocamides"] = ""
for i in 1:size(inputDB, 1)
    if (cocamidesOrNot(inputDB, i) == true)
        inputDB[i, "Cocamides"] = "yes"
        push!(trainCocamide, i)
    elseif (cocamidesOrNot(inputDB, i) == false)
        inputDB[i, "Cocamides"] = "no"
        push!(trainNonCocamide, i)
    end
end
savePath = "D:\\0_data\\dataframe_dfTrainSetWithStratification_withCNLPredictedRi_withCocamides.csv"
CSV.write(savePath, inputDB)

testCocamide = []
testNonCocamide = []
inputDB_test[!, "Cocamides"] = ""
for i in 1:size(inputDB_test, 1)
    if (cocamidesOrNot(inputDB_test, i) == true)
        inputDB_test[i, "Cocamides"] = "yes"
        push!(testCocamide, i)
    elseif (cocamidesOrNot(inputDB, i) == false)
        inputDB_test[i, "Cocamides"] = "no"
        push!(testNonCocamide, i)
    end
end
savePath = "D:\\0_data\\dataframe_dfTestSetWithStratification_withCNLPredictedRi_withCocamides.csv"
CSV.write(savePath, inputDB_test)

plotTrain = marginalkde(
        inputDB[:, end], 
        predictedRi_train, 
        xlabel = "FP-derived Ri values", 
        ylabel = "CNL-derived Ri values", 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
plot!(plotTrain.spmap[:contour], 
        inputDB[:, end] -> inputDB[:, end], c=:red, 
        label = false, 
        title = "RF Model Training With Stratification", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
scatter!(inputDB[trainCocamide, end], predictedRi_train[trainCocamide], 
        markershape = :star, 
        c = :yellow, 
        label = "Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
scatter!(inputDB[trainNonCocamide, end], predictedRi_train[trainNonCocamide], 
        markershape = :star, 
        c = :orange, 
        label = "Non-Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
        # Saving
savefig(plotTrain, "D:\\2_output\\CNLRiPrediction_RFTrainWithStratification.png")

plotTest = marginalkde(
        inputDB_test[:, end], 
        predictedRi_test, 
        xlabel = "FP-derived Ri values", 
        ylabel = "CNL-derived Ri values", 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
plot!(plotTest.spmap[:contour], 
        inputDB_test[:, end] -> inputDB_test[:, end], c=:red, 
        label = false, 
        title = "RF Model Test With Stratification", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
scatter!(inputDB_test[testCocamide, end], predictedRi_test[testCocamide], 
        markershape = :star, 
        c = :yellow, 
        label = "Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
scatter!(inputDB[testNonCocamide, end], predictedRi_test[testNonCocamide], 
        markershape = :star, 
        c = :orange, 
        label = "Non-Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
        # Saving
savefig(plotVal, "D:\\2_output\\CNLRiPrediction_RFTestWithStratification.png")