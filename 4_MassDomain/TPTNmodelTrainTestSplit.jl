VERSION
using Pkg
using CSV, DataFrames, Conda, LinearAlgebra, Statistics

# Train/Test Split by Leverage
using ProgressBars
using LinearAlgebra
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

# 516966 x 15978 df -> 16 df
#dfOutput = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_1withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_2withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_3withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_4withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_5withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_6withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_7withCNLRideltaRi.csv", DataFrame)
dfOutput = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_8withCNLRideltaRi.csv", DataFrame)

dfOutput = dfOutput[:, vcat(collect(1:13), end-2, end-1, end)]
savePath = "F:\\dataframeTPTNModeling_8.csv"
CSV.write(savePath, dfOutput)

dfOutput1 = CSV.read("F:\\dataframeTPTNModeling_1.csv", DataFrame)
dfOutput2 = CSV.read("F:\\dataframeTPTNModeling_2.csv", DataFrame)
dfOutput3 = CSV.read("F:\\dataframeTPTNModeling_3.csv", DataFrame)
dfOutput4 = CSV.read("F:\\dataframeTPTNModeling_4.csv", DataFrame)
dfOutput5 = CSV.read("F:\\dataframeTPTNModeling_5.csv", DataFrame)
dfOutput6 = CSV.read("F:\\dataframeTPTNModeling_6.csv", DataFrame)
dfOutput7 = CSV.read("F:\\dataframeTPTNModeling_7.csv", DataFrame)
dfOutput8 = CSV.read("F:\\dataframeTPTNModeling_8.csv", DataFrame)
dfOutput = vcat(dfOutput1, dfOutput2, dfOutput3, dfOutput4, dfOutput5, dfOutput6, dfOutput7, dfOutput8)
savePath = "F:\\dataframeTPTNModeling.csv"
CSV.write(savePath, dfOutput)

X = deepcopy(dfOutput[:, vcat(collect(5:12), end-1)])  # 693685 x 790 df
size(X)
Y = deepcopy(dfOutput[:, end])  #693685,
size(Y)
Xmat = Matrix(X)

# 790 x 790
hipinv = zeros(9, 9)
hipinv[:,:] .= pinv(Xmat'*Xmat)

function leverage_dist(X)   # Set x1 and x2 to your FPs variables
    h = zeros(4135721,1)
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

dfOutput[!, "GROUP"] .= ""
dfOutput[!, "Leverage"] .= float(0)
dfOutput[X_trainIdx, "GROUP"] .= "train"  # 0.8 > 3308576 333
dfOutput[X_testIdx, "GROUP"] .= "test"  # 0.2 > 827145

count = 1
for i in X_trainIdx
    dfOutput[i, "Leverage"] = train_lev[count]
    count += 1
end

count = 1
for i in X_testIdx
    dfOutput[i, "Leverage"] = test_lev[count]
    count += 1
end

# output csv is a 4135721 x 18 df
dfOutput
savePath = "F:\\dataframeTPTNModeling_withLeverage.csv"
CSV.write(savePath, dfOutput)

# 3308576 x 1
X_trainIdxDf = DataFrame([X_trainIdx], ["INDEX"])
savePath = "F:\\dataframeTPTNModeling_TrainIndex.csv"
CSV.write(savePath, X_trainIdxDf)

# 827145 x 1
X_testIdxDf = DataFrame([X_testIdx], ["INDEX"])
savePath = "F:\\dataframeTPTNModeling_TestIndex.csv"
CSV.write(savePath, X_testIdxDf)

# ==============================================================================
# train 3308576 rows
dfOutputTrain = dfOutput[dfOutput.GROUP .== "train", :]
savePath = "F:\\dataframeTPTNModeling_TrainDF.csv"
CSV.write(savePath, dfOutputTrain)

# ==============================================================================
# test 827145 rows
dfOutputTest = dfOutput[dfOutput.GROUP .== "test", :]
savePath = "F:\\dataframeTPTNModeling_TestDF.csv"
CSV.write(savePath, dfOutputTest)
describe(dfOutputTest)