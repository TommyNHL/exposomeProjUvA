VERSION
using Pkg
using CSV, DataFrames, LinearAlgebra, Statistics

# Train/Test Split by Leverage
using ProgressBars
using LinearAlgebra
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

# import 693685 x 792 df
dfOutputFP = CSV.read("F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification_Freq.csv", DataFrame)

X = deepcopy(dfOutputFP[:, 2:end-1])  # 693685 x 790 df
size(X)
Y = deepcopy(dfOutputFP[:, end])  #693685,
size(Y)
Xmat = Matrix(X)

# 790 x 790
hipinv = zeros(790, 790)
hipinv[:,:] .= pinv(Xmat'*Xmat)

function leverage_dist(X)   # Set x1 and x2 to your FPs variables
    h = zeros(693685,1)
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

function strat_split(leverage=ht; limits = limits)
    n = length(leverage)
    bin = collect(1:n)
    for i = 1: (length(limits)-1)
        bin[limits[i] .<= leverage] .= i
    end
    X_train, X_test, y_train, y_test = train_test_split(collect(1:n), leverage, test_size = 0.20, random_state = 42, stratify = bin)
    return  X_train, X_test, y_train, y_test
end

X_trainIdx, X_testIdx, train_lev, test_lev = strat_split(ht, limits = collect(0.0:0.2:1))

dfOutputFP[!, "GROUP"] .= ""
dfOutputFP[!, "Leverage"] .= float(0)
dfOutputFP[X_trainIdx, "GROUP"] .= "train"  # 0.8 > 554948
dfOutputFP[X_testIdx, "GROUP"] .= "test"  # 0.2 > 138737

count = 1
for i in X_trainIdx
    dfOutputFP[i, "Leverage"] = train_lev[count]
    count += 1
end

count = 1
for i in X_testIdx
    dfOutputFP[i, "Leverage"] = test_lev[count]
    count += 1
end

# output csv is a 693685 x 1+790+1+2 df
dfOutputFP
savePath = "F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification_FreqAndLeverage.csv"
CSV.write(savePath, dfOutputFP)

# 554948 x 1
X_trainIdxDf = DataFrame([X_trainIdx], ["INDEX"])
savePath = "F:\\UvA\\dataframe_dfTrainSetWithStratification_index.csv"
CSV.write(savePath, X_trainIdxDf)

# 138737 x 1
X_testIdxDf = DataFrame([X_testIdx], ["INDEX"])
savePath = "F:\\UvA\\dataframe_dfTestSetWithStratification_index.csv"
CSV.write(savePath, X_testIdxDf)