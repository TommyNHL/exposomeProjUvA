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
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using ScikitLearn.GridSearch: GridSearchCV

# inputing 693677 x 3+21567 df
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputTPTNdf = CSV.read("D:\\Cand_search_rr0_0612_TEST_100-400_extractedWithDeltaRi.csv", DataFrame)
sort!(inputTPTNdf, [:LABEL, :INCHIKEY_ID, :DeltaRi, :FinalScore])

inputDfOnlyTP = inputTPTNdf[inputTPTNdf.LABEL .== 1, :]
inputDfOnlyTN = inputTPTNdf[inputTPTNdf.LABEL .== 0, :]

# Train/Test Split by Leverage
## calculating how large the portion of TN is needed to be removed
X = deepcopy(inputDfOnlyTN[:, 2:end-1])  # 693677 x 790 df
size(X)
Y = deepcopy(inputDfOnlyTN[:, end])  #693677,
size(Y)

using ProgressBars
using LinearAlgebra
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split
function leverage_dist(X)   # Set x1 and x2 to your FPs variables
    h = zeros(xxx,1)
    for i in ProgressBar(1: size(X,1)) #check dimensions
        x = X[i,:] 
        hi = x'*pinv(X'*X)*x
        #push!(h,hi)
        h[i,1] = hi
    end
    return h
end

h = leverage_dist(Matrix(X))

function strat_split(leverage=h; limits = limits)
    n = length(leverage)
    bin = collect(1:n)
    for i = 1: (length(limits)-1)
        bin[limits[i].<= leverage].= i
    end
    X_retain, X_remove, y_retain, y_remove = train_test_split(collect(1:length(leverage)), leverage, test_size = 0.50, random_state = 42, stratify = bin)
    return  X_retain, X_remove, y_retain, y_remove
end

X_retainIdx, X_removeIdx, retain_lev, remove_lev = strat_split(h, limits = collect(0.0:0.2:1))
inputDfOnlyTN[!, "GROUP"] .= ""
inputDfOnlyTN[!, "Leverage"] .= float(0)
inputDfOnlyTN[X_retainIdx, "GROUP"] .= "retained"
inputDfOnlyTN[X_removeIdx, "GROUP"] .= "removed"
count = 1
for i in X_retainIdx
    inputDfOnlyTN[i, "Leverage"] = retain_lev[count]
    count += 1
end
count = 1
for i in X_removeIdx
    inputDfOnlyTN[i, "Leverage"] = remove_lev[count]
    count += 1
end

# output csv is a xxx x 3+790+2 df
savePath = "D:\\dataframe_dfTN_withLeverage.csv"
CSV.write(savePath, inputDfOnlyTN)

function create_train_test_split_strat(total_df, y_data, X_retainIdx, X_removeIdx, RiCol = true)
    #X_train_ind, X_test_ind, train_lev, test_lev = strat_split(leverage, limits = limits)
    # Create train test split of total DataFrame and dependent variables using the chosen parameters
    X_retainTN = total_df[X_retainIdx,:]
    X_removeTN = total_df[X_removeIdx,:]
    if (RiCol == true)
        Y_retainLabel = y_data[X_retainIdx]
        Y_removeLabel = y_data[X_removeIdx]
        return  X_retainTN, X_removeTN, Y_retainLabel, Y_removeLabel
    end
    # # Select train and test set of independent variables 
    # X_train = total_train[:, start_col_X_data:end]
    # X_test = total_test[:, start_col_X_data:end]
    return  X_retainTN, X_removeTN
end

X_retainTN, X_removeTN, Y_retainLabel, Y_removeLabel = create_train_test_split_strat(X, Y, X_retainIdx, X_removeIdx, true)

inputDfOnlyTN

df_info = inputDfOnlyTN[:, 1:1]
df_info
X_retainInfo, X_removeInfo = create_train_test_split_strat(df_info, df_info, X_retainIdx, X_removeIdx, false)

dfTNretainSet = hcat(X_retainInfo, X_retainTN, Y_retainLabel)
dfTNretainSet
# output csv is a 693677*0.7 x 3+22357+1 df
savePath = "D:\\dataframe_dfTNretainSet.csv"
CSV.write(savePath, dfTNretainSet)

dfTNremoveSet = hcat(X_removeInfo, X_removeTN, Y_removeLabel)
dfTNremoveSet
# output csv is a 693677*0.3 x 3+22357+1 df
savePath = "D:\\dataframe_dfTNremoveSet.csv"
CSV.write(savePath, dfTNremoveSet)

# creating new df for Train/Test
outputDf = vcat(inputDfOnlyTP, dfTNretainSet)

# outputing df
savePath = "D:\\dataframe_dfTPTNfinalSet4TrainValTest.csv"
CSV.write(savePath, outputDf)