## INPUT(S)
# dataAllFP_withNewPredictedRiWithStratification_Freq.csv

## OUTPUT(S)
# dataframe73_dfTrainSetWithStratification_95index.csv
# dataframe73_dfTestSetWithStratification_95index.csv
# dataframe73_dfWithStratification_95index.csv
# dataAllFP73_withNewPredictedRiWithStratification_FreqAnd95Leverage.csv

VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using CSV, DataFrames, LinearAlgebra, Statistics
using ProgressBars
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

## perform Train/Test Split by Leverage ##
    ## import 693685 x 792 df ##
    dfOutputFP = CSV.read("F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification_Freq.csv", DataFrame)

    ## compute leverage ##
    X = deepcopy(dfOutputFP[:, 2:end-1])  # 693685 x 790 df
    size(X)
    Y = deepcopy(dfOutputFP[:, end])  #693685,
    size(Y)
    Xmat = Matrix(X)
    #
    # 790 x 790
    hipinv = zeros(790, 790)
    hipinv[:,:] .= pinv(Xmat'*Xmat)
    #
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
    #
    h = leverage_dist(Matrix(X))
    ht = Vector(transpose(h)[1,:])
    #
function strat_split(leverage=ht; limits = limits)
    n = length(leverage)
    bin = collect(1:n)
    for i = 1: (length(limits)-1)
        bin[limits[i] .<= leverage] .= i
    end
    X_train, X_test, y_train, y_test = train_test_split(collect(1:n), leverage, test_size = 0.30, random_state = 42, stratify = bin)
    return  X_train, X_test, y_train, y_test
end
#
X_trainIdx, X_testIdx, train_lev, test_lev = strat_split(ht, limits = collect(0.0:0.2:1))
#
dfOutputFP[!, "GROUP"] .= ""
dfOutputFP[!, "Leverage"] .= float(0)
dfOutputFP[X_trainIdx, "GROUP"] .= "train"  # 0.7 > 485579
dfOutputFP[X_testIdx, "GROUP"] .= "test"  # 0.3 > 208106
#
count = 1
for i in X_trainIdx
    dfOutputFP[i, "Leverage"] = train_lev[count]
    count += 1
end
#
count = 1
for i in X_testIdx
    dfOutputFP[i, "Leverage"] = test_lev[count]
    count += 1
end
    ## apply Applicability Domain filter for the training dataset ##
    # 485579-2 x 1
    X_trainIdxArr = []
    for i in X_trainIdx
        if dfOutputFP[i, "Leverage"] <= 0.275
            push!(X_trainIdxArr, i)
        end
    end
## save index info for training ##
X_trainIdxDf = DataFrame([X_trainIdxArr], ["INDEX"])
savePath = "F:\\UvA\\dataframe73_dfTrainSetWithStratification_95index.csv"
CSV.write(savePath, X_trainIdxDf)
    ## apply Applicability Domain filter for the testing dataset ##
    # 208106-2 x 1
    X_testIdxArr = []
    for i in X_testIdx
        if dfOutputFP[i, "Leverage"] <= 0.275
            push!(X_testIdxArr, i)
        end
    end
## save index info for testing ##
X_testIdxDf = DataFrame([X_testIdxArr], ["INDEX"])
savePath = "F:\\UvA\\dataframe73_dfTestSetWithStratification_95index.csv"
CSV.write(savePath, X_testIdxDf)

## save all index info ##
X_IdxDf = vcat(X_trainIdxDf, X_testIdxDf)
sort!(X_IdxDf, [:INDEX])
savePath = "F:\\UvA\\dataframe73_dfWithStratification_95index.csv"
CSV.write(savePath, X_IdxDf)

## save CNL df after pplicability Domain filtering
# output csv is a 693685-4 x 1+790+1+2 df
dfOutputFP
dfOutputFP = dfOutputFP[dfOutputFP.Leverage .<= 0.275, :]
savePath = "F:\\UvA\\dataAllFP73_withNewPredictedRiWithStratification_FreqAnd95Leverage.csv"
CSV.write(savePath, dfOutputFP)
