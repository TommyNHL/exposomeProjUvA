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
#pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier

# inputing 693685 x 4 df
# columns: SMILES, INCHIKEY, PRECURSOR_ION, CNLmasses...
inputDB = CSV.read("D:\\0_data\\databaseOfInternal_withNLs.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES, :PRECURSOR_ION, :CNLmasses])

# inputing 693685 x 3+1+15961 df
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputCNLs = CSV.read("D:\\0_data\\dataframeCNLsRows.csv", DataFrame)
sort!(inputCNLs, [:ENTRY])

# creating a table with 2 columns
dfOutput = DataFrame([[],[]], ["INCHIKEY", "FREQUENCY"])
size(dfOutput)

count = 0
str = inputDB[1, "INCHIKEY"]
for i in 1:size(inputDB, 1)
    if (i == size(inputDB, 1))
        temp = []
        count += 1
        push!(temp, inputDB[i, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput, temp)
    elseif (inputDB[i, "INCHIKEY"] == str)
        count += 1
    else
        temp = []
        push!(temp, inputDB[i-1, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput, temp)
        str = inputDB[i, "INCHIKEY"]
        count = 1
    end
end

# 27211 x 2
dfOutput
# save
# output csv is a 27211 x 2 df
savePath = "D:\\0_data\\countingRows4Leverage.csv"
CSV.write(savePath, dfOutput)

# comparing, 30684 x 793 df
inputAllFPDB = CSV.read("D:\\0_data\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
sort!(inputAllFPDB, [:INCHIKEY, :SMILES])

# creating a table with 2 columns
dfOutput2 = DataFrame([[],[]], ["INCHIKEY", "FREQUENCY"])
size(dfOutput2)

count = 0
str = inputAllFPDB[1, "INCHIKEY"]
for i in 1:size(inputAllFPDB, 1)
    if (i == size(inputAllFPDB, 1))
        temp = []
        count += 1
        push!(temp, inputAllFPDB[i, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput2, temp)
    elseif (inputAllFPDB[i, "INCHIKEY"] == str)
        count += 1
    else
        temp = []
        push!(temp, inputAllFPDB[i-1, "INCHIKEY"])
        push!(temp, count)
        push!(dfOutput2, temp)
        str = inputAllFPDB[i, "INCHIKEY"]
        count = 1
    end
end

# 28536 x 2
dfOutput2
# save
# output csv is a 28536 x 2 df
savePath = "D:\\0_data\\countingRowsInFP4Leverage.csv"
CSV.write(savePath, dfOutput2)

# creating FP df taking frequency into accounut
# creating a table with 1+FPs+Ri columns
dfOutputFP = DataFrame([[]], ["INCHIKEY"])
for col in names(inputAllFPDB)[3:end]
    dfOutputFP[:, col] = []
end
size(dfOutputFP)  # 0 x 792

for ID in 1:size(dfOutput, 1)
    println(ID)
    for i = 1:dfOutput[ID, "FREQUENCY"]
        temp = []
        push!(temp, dfOutput[ID, "INCHIKEY"])
        rowNo = findall(inputAllFPDB.INCHIKEY .== dfOutput[ID, "INCHIKEY"])[end:end]
        for col in names(inputAllFPDB)[3:end]
            push!(temp, inputAllFPDB[rowNo, col][1])
        end
        push!(dfOutputFP, temp)
    end
end

# 693685 x 1+790+1 df
dfOutputFP
# save
# output csv is a 693685 x 792 df
savePath = "D:\\0_data\\dataAllFP_withNewPredictedRiWithStratification_Freq.csv"
CSV.write(savePath, dfOutputFP)

# Train/Test Split by Leverage
using ProgressBars
using LinearAlgebra
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

dfOutputFP = CSV.read("D:\\0_data\\dataAllFP_withNewPredictedRiWithStratification_Freq.csv", DataFrame)

X = deepcopy(dfOutputFP[:, 2:end-1])  # 693685 x 790 df
size(X)
Y = deepcopy(dfOutputFP[:, end])  #693685,
size(Y)
Xmat = Matrix(X)

# 693685 x 693685
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
    X_train, X_test, y_train, y_test = train_test_split(collect(1:n), leverage, test_size = 0.30, random_state = 42, stratify = bin)
    return  X_train, X_test, y_train, y_test
end

X_trainIdx, X_testIdx, train_lev, test_lev = strat_split(ht, limits = collect(0.0:0.2:1))

dfOutputFP[!, "GROUP"] .= ""
dfOutputFP[!, "Leverage"] .= float(0)
dfOutputFP[X_trainIdx, "GROUP"] .= "train"
dfOutputFP[X_testIdx, "GROUP"] .= "test"

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
savePath = "D:\\0_data\\dataAllFP_withNewPredictedRiWithStratification_FreqAndLeverage.csv"
CSV.write(savePath, dfOutputFP)

function create_train_test_split_strat(total_df, y_data, X_trainIdx, X_testIdx, RiCol = true)
    #X_train_ind, X_test_ind, train_lev, test_lev = strat_split(leverage, limits = limits)
    # Create train test split of total DataFrame and dependent variables using the chosen parameters
    X_trainFP = total_df[X_trainIdx,:]
    X_testFP = total_df[X_testIdx,:]
    if (RiCol == true)
        Y_trainFPRi = y_data[X_trainIdx]
        Y_testFPRi = y_data[X_testIdx]
        return  X_trainFP, X_testFP, Y_trainFPRi, Y_testFPRi
    end
    # # Select train and test set of independent variables 
    # X_train = total_train[:, start_col_X_data:end]
    # X_test = total_test[:, start_col_X_data:end]
    return  X_trainFP, X_testFP
end

X_trainFP, X_testFP, Y_trainFPRi, Y_testFPRi = create_train_test_split_strat(X, Y, X_trainIdx, X_testIdx, true)

inputCNLs
CNLs = deepcopy(inputCNLs[:, 4:end])
size(CNLs)

X_trainCNL, X_testCNL = create_train_test_split_strat(CNLs, CNLs, X_trainIdx, X_testIdx, false)

df_info = hcat(inputCNLs[:, 1:1], dfOutputFP[:,1:2])
df_info
X_trainInfo, X_testInfo = create_train_test_split_strat(df_info, df_info, X_trainIdx, X_testIdx, false)

dfTrainSetWithStratification = hcat(X_trainInfo, X_trainCNL, Y_trainFPRi)
dfTrainSetWithStratification
# output csv is a 693685*0.7 x 3+1+15961+1 df
savePath = "D:\\0_data\\dataframe_dfTrainSetWithStratification.csv"
CSV.write(savePath, dfTrainSetWithStratification)

dfTestSetWithStratification = hcat(X_testInfo, X_testCNL, Y_testFPRi)
dfTestSetWithStratification
# output csv is a 693685*0.3 x 3+1+15961+1 df
savePath = "D:\\0_data\\dataframe_dfTestSetWithStratification.csv"
CSV.write(savePath, dfTestSetWithStratification)