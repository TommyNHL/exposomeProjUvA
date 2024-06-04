VERSION
using Pkg
using CSV, DataFrames, Conda, LinearAlgebra, Statistics

# Train/Test Split by Leverage
using ProgressBars
using LinearAlgebra
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

# CNL model 95% leverage cut-off = 0.14604417882015916
# 512981 x 15978 df -> 17 df
dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_1withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_2withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_3withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_4withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_5withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_6withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_7withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_8withCNLRideltaRi.csv", DataFrame)
dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_PestwithCNLRideltaRi.csv", DataFrame)

dfOutput = dfOutput[:, vcat(collect(1:13), end-3, end-2, end-1, end)]
savePath = "F:\\UvA\\dataframeTPTNModeling_8.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_pest.csv"
CSV.write(savePath, dfOutput)

dfOutput1 = CSV.read("F:\\UvA\\dataframeTPTNModeling_1.csv", DataFrame)
dfOutput2 = CSV.read("F:\\UvA\\dataframeTPTNModeling_2.csv", DataFrame)
dfOutput3 = CSV.read("F:\\UvA\\dataframeTPTNModeling_3.csv", DataFrame)
dfOutput4 = CSV.read("F:\\UvA\\dataframeTPTNModeling_4.csv", DataFrame)
dfOutput5 = CSV.read("F:\\UvA\\dataframeTPTNModeling_5.csv", DataFrame)
dfOutput6 = CSV.read("F:\\UvA\\dataframeTPTNModeling_6.csv", DataFrame)
dfOutput7 = CSV.read("F:\\UvA\\dataframeTPTNModeling_7.csv", DataFrame)
dfOutput8 = CSV.read("F:\\UvA\\dataframeTPTNModeling_8.csv", DataFrame)
dfOutput = vcat(dfOutput1, dfOutput2, dfOutput3, dfOutput4, dfOutput5, dfOutput6, dfOutput7, dfOutput8)
savePath = "F:\\UvA\\dataframeTPTNModeling.csv"
CSV.write(savePath, dfOutput)

X = deepcopy(dfOutput[:, vcat(collect(5:12), end-1)])  # 4103848 x 9 df
size(X)
Y = deepcopy(dfOutput[:, end])  # 4103848,
size(Y)
Xmat = Matrix(X)

# 9 x 9
hipinv = zeros(9, 9)
hipinv[:,:] .= pinv(Xmat'*Xmat)

function leverage_dist(X)   # Set x1 and x2 to your FPs variables
    h = zeros(4103848,1)
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
dfOutput[X_trainIdx, "GROUP"] .= "train"  # 4103848 x 0.8 > 3283078
dfOutput[X_testIdx, "GROUP"] .= "test"  # 4103848 x 0.2 > 820770

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

# output csv is a 4103848 x 17 + 2 df
dfOutput
savePath = "F:\\UvA\\dataframeTPTNModeling_withLeverage.csv"
CSV.write(savePath, dfOutput)

# 3283078 x 1
X_trainIdxDf = DataFrame([X_trainIdx], ["INDEX"])
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainIndex.csv"
CSV.write(savePath, X_trainIdxDf)

# 820770 x 1
X_testIdxDf = DataFrame([X_testIdx], ["INDEX"])
savePath = "F:\\UvA\\dataframeTPTNModeling_TestIndex.csv"
CSV.write(savePath, X_testIdxDf)

# ==============================================================================
# train 3283078 x 17 + 2 rows
dfOutputTrain = dfOutput[dfOutput.GROUP .== "train", :]
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDF.csv"
CSV.write(savePath, dfOutputTrain)

# ==============================================================================
# test 820770 x 17 + 2 rows
dfOutputTest = dfOutput[dfOutput.GROUP .== "test", :]
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDF.csv"
CSV.write(savePath, dfOutputTest)
describe(dfOutputTest)
