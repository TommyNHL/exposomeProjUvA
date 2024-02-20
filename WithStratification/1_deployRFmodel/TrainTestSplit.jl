using Pkg
#Pkg.add("BSON")
#Pkg.add(PackageSpec(url=""))
#using BSON
using CSV, DataFrames #, PyCall, Conda, LinearAlgebra, Statistics
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#pcp = pyimport("pubchempy")
#pd = pyimport("padelpy")
#jl = pyimport("joblib")

# inputing 693677 x 4 df
# columns: SMILES, INCHIKEY, PRECURSOR_ION, CNLmasses...
inputDB = CSV.read("D:\\0_data\\databaseOfInternal_withNLs.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES, :PRECURSOR_ION, :CNLmasses])

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
        rowNo = findall(inputAllFPDB.INCHIKEY .== inputAllFPDB[i, "INCHIKEY"][end:end])
        for col in names(inputAllFPDB)[3:end]
            push!(temp, inputAllFPDB[rowNo, col])
        end
        push!(dfOutputFP, temp)
    end
end

# 27211 x 1+790+1 df
dfOutputFP
# save
# output csv is a 27211 x 792 df
savePath = "D:\\0_data\\dataAllFP_withNewPredictedRiWithStratification_Freq.csv"
CSV.write(savePath, dfOutputFP)

# Train/Test Split by Leverage
X = deepcopy(dfOutputFP)
select!(X, Not([:INCHIKEY, :predictRi]))
size(X)
Y = deepcopy(inputDB[:, end])
size(Y)

#= function leverage_dist(X)   # Set x1 and x2 to your FPs variables
    h = []
    for i in ProgressBar(1: size(X,1)) #check dimensions
        x = X[i,:] 
        hi = x'*pinv(X'*X)*x
        push!(h,hi)
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
    X_train, X_test, y_train, y_test = train_test_split(collect(1:length(leverage)), leverage, test_size = 0.9, random_state = 42, stratify = bin)
    return  X_train, X_test, y_train, y_test
end

function create_train_test_split_strat(total_df, y_data, leverage=h; limits = collect(0.0:0.2:1)) # SET START_COL_X_DATA TO 3!!!
    X_train_ind, X_test_ind, train_lev, test_lev = strat_split(leverage, limits = limits)
    # Create train test split of total DataFrame and dependent variables using the chosen parameters
    X_train = total_df[X_train_ind,:]
    X_test = total_df[X_test_ind,:]
    y_train = y_data[X_train_ind]
    y_test = y_data[X_test_ind]
    # # Select train and test set of independent variables 
    # X_train = total_train[:, start_col_X_data:end]
    # X_test = total_test[:, start_col_X_data:end]
    return  X_train, X_test, y_train, y_test, train_lev, test_lev
end

create_train_test_split_strat(X, Y, h) =#

# give 3+21567+1 = 21571 columns
trainSet, valSet = partitionTrainVal(inputDB, 0.7)  # 70% train 30% Val/Test
size(trainSet)
size(valSet)