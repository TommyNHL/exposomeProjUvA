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
Conda.add("sklearn.metrics")
## import packages ##
#using PyCall, Conda                 #using python packages
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models
sklearnmetrics = pyimport("sklearn.metrics")

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
@sk_import metrics: recall_score
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using ScikitLearn.GridSearch: GridSearchCV

# inputing 693677 x 1+8+1+1 df
# columns: "INCHIKEY_ID", "RefMatchFrag", "UsrMatchFrag", "MS1Error", "MS2Error", "MS2ErrorStd", 
    #"DirectMatch", "ReversMatch", "FinalScore", "DeltaRi", "LABEL"
#= inputTPTNdf = CSV.read("D:\\dataframe_dfTPTNfinalSet4TrainValTest", DataFrame)
sort!(inputTPTNdf, [:LABEL, :INCHIKEY_ID, :DeltaRi, :FinalScore]) =#
inputTPTNdf = CSV.read("D:\\Cand_search_rr0_0612_TEST_100-400_extractedWithDeltaRi", DataFrame)
sort!(inputTPTNdf, [:LABEL, :INCHIKEY_ID, :DeltaRi, :FinalScore])

# Train/Test Split by Leverage
using ProgressBars
using LinearAlgebra
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

X = deepcopy(inputTPTNdf[:, 2:end-1])  # 693677 x 790 df
size(X)
Y = deepcopy(inputTPTNdf[:, end])  #693677,
size(Y)
Xmat = Matrix(X)

# 9 x 9
hipinv = zeros(9, 9)
hipinv[:,:] .= pinv(Xmat'*Xmat)

function leverage_dist(X)   # Set x1 and x2 to your FPs variables
    h = zeros(xxx,1)
    for i in ProgressBar(1: size(X,1)) #check dimensions
        x = X[i,:] 
        #hi = x'*pinv(X'*X)*x
        hi = x'* hipinv *x
        #push!(h,hi)
        h[i,1] = hi
    end
    return h
end

h = leverage_dist(Matrix(X))
ht = Vector(transpose(h)[1,:])

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
inputTPTNdf[X_trainIdx, "GROUP"] .= "train"  # 0.7 > 485579
inputTPTNdf[X_testIdx, "GROUP"] .= "test"  # 0.3 > 208106

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

# output csv is a 693677 x 1+9+1+2 df
inputTPTNdf
savePath = "D:\\dataframe_dfTPTNfinalSet4TrainValTest_withLeverage.csv"
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

inputTPTNdf

X_trainTPTN, X_testTPTN, Y_trainLabel, Y_testLabel = create_train_test_split_strat(X, Y, X_trainIdx, X_testIdx, true)

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

# preparing Matrix for ML
X_inputDB_test = deepcopy(dfTestSet[:, 2:end-1])
size(X_inputDB_test)
Y_inputDB_test = deepcopy(dfTestSet[:, end])
size(Y_inputDB_test)

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
function errorDetermination(arrY, predictedY)
    sumAE = 0
    maxAE = 0
    for i = 1:size(predictedY, 1)
        AE = abs(arrY[i] - predictedY[i])
        if (AE > maxAE)
            maxAE = AE
        end
        sumAE += (AE ^ 2)
    end
    MSE = sumAE / size(predictedY, 1)
    RMSE = MSE ^ 0.5
    return maxAE, MSE, RMSE
end

## R-square value
function rSquareDetermination(arrY, predictedY)
    sumY = 0
    for i = 1:size(predictedY, 1)
        sumY += predictedY[i]
    end
    meanY = sumY / size(predictedY, 1)
    sumAE = 0
    sumRE = 0
    for i = 1:size(predictedY, 1)
        AE = abs(arrY[i] - predictedY[i])
        RE = abs(arrY[i] - meanY)
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
function optimRandomForestClassifier(df_train, df_test)
    #leaf_r = [collect(4:2:10);15;20]
    leaf_r = collect(8:8:32)
    #tree_r = vcat(collect(50:50:400),collect(500:100:1000))
    tree_r = collect(50:50:300)
    z = zeros(1,11)
    itr = 1
    for l in leaf_r
        for t in tree_r
            for state = 1:3
                println("itr=", itr, ", leaf=", l, ", tree=", t, ", s=", state)
                MaxFeat = Int64((ceil(size(df_train,2)-1)/3))
                println("## split ##")
                M_train, M_val = partitionTrainVal(df_train, 0.67)
                Xx_train = deepcopy(M_train[:, 2:end-1])
                Yy_train = deepcopy(M_train[:, end])
                Xx_val = deepcopy(M_val[:, 2:end-1])
                Yy_val = deepcopy(M_val[:, end])
                xx_test = deepcopy(df_test[:, 2:end-1])
                yy_test = deepcopy(df_test[:, end])
                println("## Regression ##")
                reg = RandomForestClassifier(class_weight={0: 2, 1: 1}, n_estimators=t, min_samples_leaf=l, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=42)
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                if itr == 1
                    z[1,1] = l
                    z[1,2] = t
                    z[1,3] = state
                    z[1,4] = score(reg, Matrix(Xx_train), Vector(Yy_train))
                    z[1,5] = score(reg, Matrix(df_train[:, 2:end-1]), Vector(df_train[:, end]))
                    println("## CV ##")
                    acc5_train = cross_val_score(reg, Matrix(df_train[:, 2:end-1]), Vector(df_train[:, end]); cv = 3)
                    z[1,5] = avgAcc(acc5_train, 3)
                    z[1,6] = score(reg, Matrix(Xx_val), Vector(Yy_val))
                    z[1,7] = score(reg, Matrix(xx_test), Vector(yy_test))
                    z[1,8] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                    z[1,9] = f1_score(Vector(yy_test), predict(reg, Matrix(xx_test)))
                    z[1,10] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                    z[1,11] = matthews_corrcoef(Vector(yy_test), predict(reg, Matrix(xx_test)))
                    println(z)
                else
                    println("## CV ##")
                    itrain= score(reg, Matrix(Xx_train), Vector(Yy_train)) 
                    traintrain = score(reg, Matrix(df_train[:, 2:end-1]), Vector(df_train[:, end]))
                    acc5_train = cross_val_score(reg, Matrix(df_train[:, 2:end-1]), Vector(df_train[:, end]); cv = 3)
                    traincvtrain = avgAcc(acc5_train, 3) 
                    ival = score(reg, Matrix(Xx_val), Vector(Yy_val)) 
                    etest = score(reg, Matrix(xx_test), Vector(yy_test))
                    f1Val = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                    f1Test = f1_score(Vector(yy_test), predict(reg, Matrix(xx_test)))
                    mccVal = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
                    mccTest = matthews_corrcoef(Vector(yy_test), predict(reg, Matrix(xx_test)))
                    z = vcat(z, [l t state itrain traintrain traincvtrain ival etest f1Val f1Test mccVal mccTest])
                    println(z)
                end
                #println("End of $itr iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], state=z[:,3], accuracy_3Ftrain = z[:,4], accuracy_train = z[:,5], avgAccuracy3FCV_train = z[:,6], accuracy_val = z[:,7], F1_val = z[:,8] F1_test = z[:,9] MCC_val = z[:,10] MCC_test = z[:,11])
    z_df_sorted = sort(z_df, [:MCC_test, :F1_test, :MCC_val, :F1_val], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestClassifier(dfTrainSet, dfTestSet)

# save, ouputing 72 x 8 df
savePath = "D:\\hyperparameterTuning_dfTPTN.csv"
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

model = RandomForestClassifier(
      class_weight={0: 2, 1: 1}, 
      n_estimators = 300, 
      #max_depth = 10, 
      min_samples_leaf = 8, 
      max_features = Int64(ceil(size(x_train,2)/3)), 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42
      )
fit!(model, Matrix(dfTrainSet[:, 2:end-1]), Vector(dfTrainSet[:, end]))

# saving model
modelSavePath = "D:\\RFmodel4TPTN.joblib"
jl.dump(model, modelSavePath, compress = 5)

# training performace, CNL-predictedRi vs. FP-predictedRi
predictedY_train = predict(model, Matrix(dfTrainSet[:, 2:end-1]))
dfTrainSet[!, "LABEL"] = predictedY_train
pTP_train = predict_proba(model, Matrix(dfTrainSet[:, 2:end-1]))
f1_train = f1_score(Vector(dfTrainSet[:, end]), predictedY_train)
mcc_train = matthews_corrcoef(Vector(dfTrainSet[:, end]), predictedY_train)
# save, ouputing trainSet df 0.7 x (3+15994+1)
savePath = "D:\\dataframe_TPTNdfTrainSet_withPredictedY.csv"
CSV.write(savePath, dfTrainSet)

maxAE_train, MSE_train, RMSE_train = errorDetermination(dfTrainSet[:, end], predictedY_train)
rSquare_train = rSquareDetermination(dfTrainSet[:, end], predictedY_train)
## accuracy
acc1_train = score(model, Matrix(dfTrainSet[:, 2:end-1]), Vector(dfTrainSet[:, end]))
acc5_train = cross_val_score(model, Matrix(dfTrainSet[:, 2:end-1]), Vector(dfTrainSet[:, end]); cv = 5)
avgAcc_train = avgAcc(acc5_train, 5)

# model validation
#load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("D:\\RFmodel4TPTN.joblib")
size(modelRF_TPTN)

predictedY_test = predict(modelRF_TPTN, Matrix(dfTestSet[:, 2:end-1]))
dfTestSet[!, "LABEL"] = predictedY_test
pTP_test = predict_proba(modelRF_TPTN, Matrix(dfTestSet[:, 2:end-1]))
f1_test = f1_score(Vector(dfTestSet[:, end]), predictedY_test)
mcc_test = matthews_corrcoef(Vector(dfTestSet[:, end]), predictedY_test)
# save, ouputing testSet df 0.3 x (3+15994+1)
savePath = "D:\\dataframe_TPTNdfTestSet_withPredictedY.csv"
CSV.write(savePath, dfTestSet)

maxAE_val, MSE_val, RMSE_val = errorDetermination(dfTestSet[:, end], predictedY_test)
rSquare_val = rSquareDetermination(dfTestSet[:, end], predictedY_test)
## accuracy
acc1_val = score(modelRF_TPTN, Matrix(dfTestSet[:, 2:end-1]), Vector(dfTestSet[:, end]))
acc5_val = cross_val_score(modelRF_TPTN, Matrix(dfTestSet[:, 2:end-1]), Vector(dfTestSet[:, end]); cv = 5)
avgAcc_val = avgAcc(acc5_val, 5)

# plots
# inputing dfs for separation of the cocamides and non-cocamides datasets
## 5364 x 931 df 
inputCocamidesTrain = CSV.read("D:\\0_data\\CocamideExtWithStartification_Fingerprints_train.csv", DataFrame)
sort!(inputCocamidesTrain, :SMILES)

## 947 x 931 df
inputCocamidesTest = CSV.read("D:\\0_data\\CocamideExtWithStartification_Fingerprints_test.csv", DataFrame)
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
dfTrainSet[!, "Cocamides"] = ""
for i in 1:size(dfTrainSet, 1)
    if (cocamidesOrNot(dfTrainSet, i) == true)
        dfTrainSet[i, "Cocamides"] = "yes"
        push!(trainCocamide, i)
    elseif (cocamidesOrNot(dfTrainSet, i) == false)
        dfTrainSet[i, "Cocamides"] = "no"
        push!(trainNonCocamide, i)
    end
end
savePath = "D:\\dataframe_TPTNdfTrainSet_withPredictedY_withCocamides.csv"
CSV.write(savePath, dfTrainSet)

testCocamide = []
testNonCocamide = []
dfTestSet[!, "Cocamides"] = ""
for i in 1:size(dfTestSet, 1)
    if (cocamidesOrNot(dfTestSet, i) == true)
        dfTestSet[i, "Cocamides"] = "yes"
        push!(testCocamide, i)
    elseif (cocamidesOrNot(dfTestSet, i) == false)
        dfTestSet[i, "Cocamides"] = "no"
        push!(testNonCocamide, i)
    end
end
savePath = "D:\\dataframe_TPTNdfTestSet_withPredictedY_withCocamides.csv"
CSV.write(savePath, dfTestSet)

plotTrain = marginalkde(
        dfTrainSet[:, end], 
        predictedY_train, 
        xlabel = "Ground-Truth Labels", 
        ylabel = "Predicted Labels", 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
plot!(plotTrain.spmap[:contour], 
        dfTrainSet[:, end] -> dfTrainSet[:, end], c=:red, 
        label = false, 
        title = "RF Model Training", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
scatter!(dfTrainSet[trainCocamide, end], predictedY_train[trainCocamide], 
        markershape = :star, 
        c = :yellow, 
        label = "Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
scatter!(dfTrainSet[trainNonCocamide, end], predictedY_train[trainNonCocamide], 
        markershape = :star, 
        c = :orange, 
        label = "Non-Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
        # Saving
savefig(plotTrain, "D:\\TPTNPrediction_RFTrain.png")

plotTest = marginalkde(
        dfTestSet[:, end], 
        predictedY_test, 
        xlabel = "Ground-Truth Labels", 
        ylabel = "Predicted Labels", 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
plot!(plotTest.spmap[:contour], 
        dfTestSet[:, end] -> dfTestSet[:, end], c=:red, 
        label = false, 
        title = "RF Model Test", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
scatter!(dfTestSet[testCocamide, end], predictedY_test[testCocamide], 
        markershape = :star, 
        c = :yellow, 
        label = "Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
scatter!(dfTestSet[testNonCocamide, end], predictedY_test[testNonCocamide], 
        markershape = :star, 
        c = :orange, 
        label = "Non-Cocamides", 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
        # Saving
savefig(plotVal, "D:\\TPTNPrediction_RFTest.png")