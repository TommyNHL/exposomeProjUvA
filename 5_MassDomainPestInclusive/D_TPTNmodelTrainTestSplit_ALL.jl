VERSION
using Pkg
using CSV, DataFrames, Conda, LinearAlgebra, Statistics

# Train/Test Split by Leverage
using ProgressBars
using LinearAlgebra
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

# CNL model 95% leverage cut-off = 0.14604417882015916
# 512981 / 136678 x 18 df
dfOutput1 = CSV.read("F:\\UvA\\dataframeTPTNModeling_1.csv", DataFrame)
dfOutput2 = CSV.read("F:\\UvA\\dataframeTPTNModeling_2.csv", DataFrame)
dfOutput3 = CSV.read("F:\\UvA\\dataframeTPTNModeling_3.csv", DataFrame)
dfOutput4 = CSV.read("F:\\UvA\\dataframeTPTNModeling_4.csv", DataFrame)
dfOutput5 = CSV.read("F:\\UvA\\dataframeTPTNModeling_5.csv", DataFrame)
dfOutput6 = CSV.read("F:\\UvA\\dataframeTPTNModeling_6.csv", DataFrame)
dfOutput7 = CSV.read("F:\\UvA\\dataframeTPTNModeling_7.csv", DataFrame)
dfOutput8 = CSV.read("F:\\UvA\\dataframeTPTNModeling_8.csv", DataFrame)
dfOutputPest = CSV.read("F:\\UvA\\dataframeTPTNModeling_pest.csv", DataFrame)
dfOutput = vcat(dfOutput1, dfOutput2, dfOutput3, dfOutput4, dfOutput5, dfOutput6, dfOutput7, dfOutput8, dfOutputPest)
savePath = "F:\\UvA\\dataframeTPTNModeling_all.csv"
CSV.write(savePath, dfOutput)  # 4240526 x 18


describe(dfOutput)[end-2:end, :]

h = deepcopy(dfOutput[:, end])
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
dfOutput[!, "LeverageOfLeverage"] .= float(0)
dfOutput[X_trainIdx, "GROUP"] .= "train"  # 4103848 x 0.8 > 3283078
dfOutput[X_testIdx, "GROUP"] .= "test"  # 4103848 x 0.2 > 820770
dfOutput[!, "IncludeDeltaRI"] .= ""

count = 1
X_trainIdxYes = []
X_trainIdxNo = []
for i in X_trainIdx
    dfOutput[i, "LeverageOfLeverage"] = train_lev[count]
    if dfOutput[i, "Leverage"] > 0.14604417882015916
        dfOutput[i, "IncludeDeltaRI"] = "no"
        push!(X_trainIdxNo, i)
    elseif dfOutput[i, "Leverage"] <= 0.14604417882015916
        dfOutput[i, "IncludeDeltaRI"] = "yes"
        push!(X_trainIdxYes, i)
    end
    count += 1
end

count = 1
X_testIdxYes = []
X_testIdxNo = []
for i in X_testIdx
    dfOutput[i, "LeverageOfLeverage"] = test_lev[count]
    if dfOutput[i, "Leverage"] > 0.14604417882015916
        dfOutput[i, "IncludeDeltaRI"] = "no"
        push!(X_testIdxNo, i)
    elseif dfOutput[i, "Leverage"] <= 0.14604417882015916
        dfOutput[i, "IncludeDeltaRI"] = "yes"
        push!(X_testIdxYes, i)
    end
    count += 1
end

# output csv is a 4240526 x 18 + 3 df
dfOutput
savePath = "F:\\UvA\\dataframeTPTNModeling_withLeverage_all.csv"
CSV.write(savePath, dfOutput)

# ==============================================================================
# 3392420 x 1
X_trainIdxDf = DataFrame([X_trainIdx], ["INDEX"])
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainIndex_all.csv"
CSV.write(savePath, X_trainIdxDf)

# train 3392420 x 18 + 3 rows
dfOutputTrain = dfOutput[dfOutput.GROUP .== "train", :]
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl_all.csv"
CSV.write(savePath, dfOutputTrain)

# 3391333 x 1
X_trainIdxDf_YesDeltaRI = DataFrame([X_trainIdxYes], ["INDEX"])
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainYesIndex_all.csv"
CSV.write(savePath, X_trainIdxDf_YesDeltaRI)

# train 3391333 x 18 + 3 rows
dfOutputTrainYes = dfOutputTrain[dfOutputTrain.IncludeDeltaRI .== "yes", :]
savePath = "F:\\UvA\\dataframeTPTNModeling_TrainYesDFwithhl_all.csv"
CSV.write(savePath, dfOutputTrainYes)

# ==============================================================================
# 848106 x 1
X_testIdxDf = DataFrame([X_testIdx], ["INDEX"])
savePath = "F:\\UvA\\dataframeTPTNModeling_TestIndex_all.csv"
CSV.write(savePath, X_testIdxDf)

# train 848106 x 18 + 3 rows
dfOutputTest = dfOutput[dfOutput.GROUP .== "test", :]
savePath = "F:\\UvA\\dataframeTPTNModeling_TestDFwithhl_all.csv"
CSV.write(savePath, dfOutputTest)

# 847838 x 1
X_testIdxDf_YesDeltaRI = DataFrame([X_testIdxYes], ["INDEX"])
savePath = "F:\\UvA\\dataframeTPTNModeling_TestYesIndex_all.csv"
CSV.write(savePath, X_testIdxDf_YesDeltaRI)

# train 847838 x 18 + 3 rows
dfOutputTestYes = dfOutputTest[dfOutputTest.IncludeDeltaRI .== "yes", :]
savePath = "F:\\UvA\\dataframeTPTNModeling_TestYesDFwithhl_all.csv"
CSV.write(savePath, dfOutputTestYes)

describe(dfOutputTest)
