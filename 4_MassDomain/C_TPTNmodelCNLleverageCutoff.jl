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
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_1withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_2withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_3withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_4withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_5withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_6withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_7withCNLRideltaRi.csv", DataFrame)
dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_8withCNLRideltaRi.csv", DataFrame)
#dfOutput = CSV.read("F:\\UvA\\dataframeCNLsRows4TPTNModeling_PestwithCNLRideltaRi.csv", DataFrame)

describe(dfOutput)[end-4:end, :]

X = deepcopy(dfOutput[:, 13:end-4])  # 512981 x 15962 df / 136678 x 15962 df
size(X)
Y = deepcopy(dfOutput[:, end-2])  #512981, 136678
size(Y)
Xmat = Matrix(X)

# 15962 x 15962
hipinv = zeros(15962, 15962)
hipinv[:,:] .= pinv(Xmat'*Xmat)

function leverage_dist(X)   # Set x1 and x2 to your FPs variables
    h = zeros(512981,1)
    #h = zeros(136678,1)
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

dfOutput[!, "Leverage"] .= ht

dfOutput = dfOutput[:, vcat(collect(1:13), end-4, end-3, end-2, end-1, end)]

# 512981 * 18 df / 136678
#savePath = "F:\\UvA\\dataframeTPTNModeling_1.csv"
#savePath = "F:\\UvA\\dataframeTPTNModeling_2.csv"
#savePath = "F:\\UvA\\dataframeTPTNModeling_3.csv"
#savePath = "F:\\UvA\\dataframeTPTNModeling_4.csv"
#savePath = "F:\\UvA\\dataframeTPTNModeling_5.csv"
#savePath = "F:\\UvA\\dataframeTPTNModeling_6.csv"
#savePath = "F:\\UvA\\dataframeTPTNModeling_7.csv"
savePath = "F:\\UvA\\dataframeTPTNModeling_8.csv"
#savePath = "F:\\UvA\\dataframeTPTNModeling_pest.csv"
CSV.write(savePath, dfOutput)
