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
dfOutput = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_4withCNLRideltaRi.csv", DataFrame)
#dfCNLs5 = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_5withCNLRideltaRi.csv", DataFrame)
#dfCNLs6 = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_6withCNLRideltaRi.csv", DataFrame)
#dfCNLs7 = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_7withCNLRideltaRi.csv", DataFrame)
#dfCNLs8 = CSV.read("F:\\dataframeCNLsRows4TPTNModeling_8withCNLRideltaRi.csv", DataFrame)

dfOutput = dfOutput[:, vcat(collect(1:13), end-2, end-1, end)]
savePath = "F:\\dataframeTPTNModeling_4.csv"
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


X = deepcopy(dfOutput[:, vcat(collect(4:11), end-1)])  # 693685 x 790 df
size(X)
Y = deepcopy(dfOutput[:, end])  #693685,
size(Y)
Xmat = Matrix(X)

# 790 x 790
hipinv = zeros(9, 9)
hipinv[:,:] .= pinv(Xmat'*Xmat)

function leverage_dist(X)   # Set x1 and x2 to your FPs variables
    h = zeros(103778,1)
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
    X_train, X_test, y_train, y_test = train_test_split(collect(1:n), leverage, test_size = 0.50, random_state = 42, stratify = bin)
    return  X_train, X_test, y_train, y_test
end

X_trainIdx, X_testIdx, train_lev, test_lev = strat_split(ht, limits = collect(0.0:0.2:1))

dfOutputFP[!, "GROUP"] .= ""
dfOutputFP[!, "Leverage"] .= float(0)
dfOutputFP[X_trainIdx, "GROUP"] .= "train"  # 0.7 > 485579
dfOutputFP[X_testIdx, "GROUP"] .= "test"  # 0.3 > 208106

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
savePath = "F:\\dataAllFP_withNewPredictedRiWithStratification_FreqAndLeverage.csv"
CSV.write(savePath, dfOutputFP)

# 485579 x 1
X_trainIdxDf = DataFrame([X_trainIdx], ["INDEX"])
savePath = "F:\\dataframe_dfTrainSetWithStratification_index.csv"
CSV.write(savePath, X_trainIdxDf)

# 208106 x 1
X_testIdxDf = DataFrame([X_testIdx], ["INDEX"])
savePath = "F:\\dataframe_dfTestSetWithStratification_index.csv"
CSV.write(savePath, X_testIdxDf)



dfCNLsSum1 = CSV.read("F:\\dfCNLsSum_1.csv", DataFrame)
dfCNLsSum2 = CSV.read("F:\\dfCNLsSum_2.csv", DataFrame)
dfCNLsSum3 = CSV.read("F:\\dfCNLsSum_3.csv", DataFrame)
dfCNLsSum4 = CSV.read("F:\\dfCNLsSum_4.csv", DataFrame)
dfCNLsSum5 = CSV.read("F:\\dfCNLsSum_5.csv", DataFrame)
dfCNLsSum6 = CSV.read("F:\\dfCNLsSum_6.csv", DataFrame)
dfCNLsSum7 = CSV.read("F:\\dfCNLsSum_7.csv", DataFrame)
dfCNLsSum8 = CSV.read("F:\\dfCNLsSum_8.csv", DataFrame)
dfCNLsSum = vcat(dfCNLsSum1, dfCNLsSum2, dfCNLsSum3, dfCNLsSum4, dfCNLsSum5, dfCNLsSum6, dfCNLsSum7, dfCNLsSum8)

sumUp = []
push!(sumUp, 8888888)
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
push!(sumUp, 8888888)
for col in names(dfCNLsSum)[14:end-1]
    count = 0
    for i in 1:size(dfCNLsSum, 1)
        count += dfCNLsSum[i, col]
    end
    push!(sumUp, count)
end
push!(sumUp, 8888888)
push!(dfCNLsSum, sumUp)
# 1076799 -> 1076800 rows
dfCNLsSum = dfCNLsSum[end:end, :]
savePath = "F:\\dfCNLsSum.csv"
CSV.write(savePath, dfCNLsSum)

using DataSci4Chem
massesCNLsDistrution = bar(names(dfCNLsSum)[14:end-1], Vector(dfCNLsSum[end, 14:end-1]), 
    label = false, 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    xlabel!("CNLs features")
    ylabel!("Summation")
    # Saving
    savefig(massesCNLsDistrution, "F:\\TPTNmassesCNLsDistrution.png")