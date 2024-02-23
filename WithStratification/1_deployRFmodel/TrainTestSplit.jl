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

# inputing 693685 x 1 df
CNLsInfo = CSV.read("F:\\databaseOfInternal_withInfoOnly.csv", DataFrame)

# inputing a 693685 x 1 df
CNLsInfo2 = CSV.read("F:\\databaseOfInternal_withInfoIsotopicMassOnly.csv", DataFrame)

# inputing 693685 x 1+15961 df
CNLs = CSV.read("F:\\databaseOfInternal_withNLsOnly.csv", DataFrame)

# inputing 693685 x 1 df
CNLsY = CSV.read("F:\\databaseOfInternal_withYOnly.csv", DataFrame)

### inputing Index for Train/Test Split
# 485579 x 1
X_trainIdxDf = CSV.read("F:\\dataframe_dfTrainSetWithStratification_index.csv", DataFrame)
X_trainIdx = X_trainIdxDf[:, "INDEX"]

# 208106 x 1
X_testIdxDf = CSV.read("F:\\dataframe_dfTestSetWithStratification_index.csv", DataFrame)
X_testIdx = X_testIdxDf[:, "INDEX"]



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

X_trainInfo, X_testInfo = create_train_test_split_strat(CNLsInfo, CNLsInfo, X_trainIdx, X_testIdx, false)

X_trainInfo2, X_testInfo2 = create_train_test_split_strat(CNLsInfo2, CNLsInfo2, X_trainIdx, X_testIdx, false)

X_trainCNL, X_testCNL = create_train_test_split_strat(CNLs, CNLs, X_trainIdx, X_testIdx, false)

Y_trainFPRi, Y_testFPRi = create_train_test_split_strat(CNLsY, CNLsY, X_trainIdx, X_testIdx, false)


dfTrainSetWithStratification = hcat(X_trainInfo, X_trainInfo2, X_trainCNL, Y_trainFPRi)
dfTrainSetWithStratification
# output csv is a 693685*0.7 x 3+1+15961+1 df = 485579 x 15966
savePath = "F:\\dataframe_dfTrainSetWithStratification.csv"
CSV.write(savePath, dfTrainSetWithStratification)

dfTestSetWithStratification = hcat(X_testInfo, X_testInfo2, X_testCNL, Y_testFPRi)
dfTestSetWithStratification
# output csv is a 693685*0.3 x 3+1+15961+1 df = 208106 x 15966
savePath = "F:\\dataframe_dfTestSetWithStratification.csv"
CSV.write(savePath, dfTestSetWithStratification)