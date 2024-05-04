using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))
using Random
using CSV, DataFrames

# inputing 693685-4 x 1 df
X_IdxDf = CSV.read("F:\\UvA\\dataframe_dfWithStratification_95index.csv", DataFrame)

# inputing 693685-4 x 1 df
CNLsInfo = CSV.read("F:\\UvA\\databaseOfInternal_withEntryInfoOnly.csv", DataFrame)
CNLsInfo = CNLsInfo[X_IdxDf[:,1], :]

# inputing a 693685-4 x 1 df
CNLsInfo2 = CSV.read("F:\\UvA\\databaseOfInternal_withINCHIKEYInfoOnly.csv", DataFrame)
CNLsInfo2 = CNLsInfo2[X_IdxDf[:,1], :]

# inputing 693685-4 x 1+15961 df
CNLs = CSV.read("F:\\UvA\\databaseOfInternal_withNLsOnly.csv", DataFrame)
CNLs = CNLs[X_IdxDf[:,1], :]

# inputing 693685-4 x 1 df
CNLsY = CSV.read("F:\\UvA\\databaseOfInternal_withYOnly.csv", DataFrame)
CNLsY = CNLsY[X_IdxDf[:,1], :]

### inputing Index for Train/Test Split
# 346842-2 x 1
X_trainIdxDf = CSV.read("F:\\UvA\\dataframe_dfTrainSetWithStratification_95index.csv", DataFrame)
X_trainIdx = X_trainIdxDf[:, "INDEX"]

# 346843-2 x 1
X_testIdxDf = CSV.read("F:\\UvA\\dataframe_dfTestSetWithStratification_95index.csv", DataFrame)
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
# output csv is a 693685*0.5 x 1+1+1+15961+1 df = 346842-2 x 15965
savePath = "F:\\UvA\\dataframe_95dfTrainSetWithStratification.csv"
CSV.write(savePath, dfTrainSetWithStratification)

dfTestSetWithStratification = hcat(X_testInfo, X_testInfo2, X_testCNL, Y_testFPRi)
dfTestSetWithStratification
# output csv is a 693685*0.5 x 1+1+1+15961+1 df = 346843-2 x 15965
savePath = "F:\\UvA\\dataframe_95dfTestSetWithStratification.csv"
CSV.write(savePath, dfTestSetWithStratification)