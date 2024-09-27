## INPUT(S)
# databaseOfInternal_withEntryInfoOnly.csv
# databaseOfInternal_withINCHIKEYInfoOnly.csv
# databaseOfInternal_withNLsOnly.csv
# databaseOfInternal_withYOnly.csv
# dataframe73_dfTrainSetWithStratification_95index.csv
# dataframe73_dfTestSetWithStratification_95index.csv

## OUTPUT(S)
# dataframe73_95dfTrainSetWithStratification.csv
# dataframe73_95dfTestSetWithStratification.csv

## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using Random
using CSV, DataFrames

## input 485577 x 1 df ##
CNLsInfo = CSV.read("F:\\UvA\\databaseOfInternal_withEntryInfoOnly.csv", DataFrame)

## input a 485577 x 1 df ##
CNLsInfo2 = CSV.read("F:\\UvA\\databaseOfInternal_withINCHIKEYInfoOnly.csv", DataFrame)

## input 485577 x 1+15961 df ##
CNLs = CSV.read("F:\\UvA\\databaseOfInternal_withNLsOnly.csv", DataFrame)

## input 485577 x 1 df ##
CNLsY = CSV.read("F:\\UvA\\databaseOfInternal_withYOnly.csv", DataFrame)

## input Index for Train/Test Split ##
# 485577 x 1
X_trainIdxDf = CSV.read("F:\\UvA\\dataframe73_dfTrainSetWithStratification_95index.csv", DataFrame)
X_trainIdx = X_trainIdxDf[:, "INDEX"]
# 208104 x 1
X_testIdxDf = CSV.read("F:\\UvA\\dataframe73_dfTestSetWithStratification_95index.csv", DataFrame)
X_testIdx = X_testIdxDf[:, "INDEX"]

## define a function for train/test split ##
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
#
X_trainInfo, X_testInfo = create_train_test_split_strat(CNLsInfo, CNLsInfo, X_trainIdx, X_testIdx, false)
X_trainInfo2, X_testInfo2 = create_train_test_split_strat(CNLsInfo2, CNLsInfo2, X_trainIdx, X_testIdx, false)
X_trainCNL, X_testCNL = create_train_test_split_strat(CNLs, CNLs, X_trainIdx, X_testIdx, false)
Y_trainFPRi, Y_testFPRi = create_train_test_split_strat(CNLsY, CNLsY, X_trainIdx, X_testIdx, false)

## save the output training set as a spreadsheet ##
dfTrainSetWithStratification = hcat(X_trainInfo, X_trainInfo2, X_trainCNL, Y_trainFPRi)
dfTrainSetWithStratification
# csv is a 693685*0.7 x 1+1+1+15961+1 df = 485577 x 15965
savePath = "F:\\UvA\\dataframe73_95dfTrainSetWithStratification.csv"
CSV.write(savePath, dfTrainSetWithStratification)

## save the output testing set as a spreadsheet ##
dfTestSetWithStratification = hcat(X_testInfo, X_testInfo2, X_testCNL, Y_testFPRi)
dfTestSetWithStratification
# output csv is a 693685*0.3 x 1+1+1+15961+1 df = 208104 x 15965
savePath = "F:\\UvA\\dataframe73_95dfTestSetWithStratification.csv"
CSV.write(savePath, dfTestSetWithStratification)
