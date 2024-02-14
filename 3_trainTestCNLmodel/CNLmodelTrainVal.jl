VERSION
using Pkg
#Pkg.add("PyCall")
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
#Pkg.add("JLD")
#Pkg.add("HDF5")
#Pkg.add("PyCallJLD")
#Pkg.add(Pkg.PackageSpec(;name="ScikitLearn", version="1.3.1"))
#using JLD, HDF5, PyCallJLD
#Pkg.add(PackageSpec(url=""))
#Pkg.add("MLDataUtils")
#Pkg.add(PackageSpec(url=""))
#using BSON
using Random
using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#using PyCall, Conda                 #using python packages
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models

using ScikitLearn: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
using ScikitLearn.GridSearch: RandomizedSearchCV

# inputing 28302 x (2+15977+1)
# columns: SMILES, INCHIKEY, CNLs, predictRi
inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOnlyCocamides.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES])

#= inputDB1 = CSV.read("D:\\0_data\\databaseOfAllMS2_withMergedNLs.csv", DataFrame)
sort!(inputDB1, [:INCHIKEY, :SMILES])

inputDB2 = CSV.read("D:\\0_data\\dataAllFP_withNewPredictedRi.csv", DataFrame)
sort!(inputDB2, [:INCHIKEY, :SMILES])

input1 = CSV.read("D:\\0_data\\CocamideExt_Fingerprints_train.csv", DataFrame)
sort!(input1, :SMILES)

input2 = CSV.read("D:\\0_data\\CocamideExt_Fingerprints_test.csv", DataFrame)
sort!(input2, :SMILES) =#

#= function existOrNot(i)
  if (inputDB[i, "INCHIKEY"] in Array(input1[:, "SMILES"]))
      return true
  else
      if (inputDB[i, "SMILES"] in Array(input2[:, "SMILES"]))
          return true
      end
  end
end =#

#= count = 0
record = []
for i in 1:size(inputDB, 1)
  if (existOrNot(i) == true)
      push!(record, i)
      count += 1
      println(count)
  end
end
for i in record
    println(inputDB[i, "SMILES"], 
        inputDB2[findall(inputDB2.SMILES .== inputDB1[i, "SMILES"]), "predictRi"], 
        input1[findall(input1.SMILES .== inputDB1[i, "SMILES"]), "RI"], 
        input2[findall(input2.SMILES .== inputDB1[i, "SMILES"]), "RI"])
end =#

function partitionTrainVal(data, at = 0.7)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

# give 2+15979+1 = 15980 columns
trainSet, valSet = partitionTrainVal(inputDB, 0.7) # 70% train
size(trainSet)
size(valSet)

# ouputing trainSet df 0.7 x (2+15977+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamidesTrain.csv"
CSV.write(savePath, trainSet)

# ouputing trainSet df 0.3 x (2+15977+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamidesVal.csv"
CSV.write(savePath, valSet)

x = deepcopy(inputDB)
x_test = deepcopy(trainSet)
y_train = deepcopy(valSet)

x_test = reshape(x_test, 1,length(x_test))

mod = RandomForestRegressor()
param_dist = Dict("n_estimators"=>[50 , 100, 200, 300],
                  "max_depth"=> [3, 5, 6 ,8 , 9 ,10])
model = RandomizedSearchCV(mod, param_dist, n_iter=10, cv=5)

fit!(model, Matrix(x), Matrix(DataFrames.dropmissing(y_train)))

predict(model, x_test)

 # Create a random forest model
model = RandomForestClassifier(n_subfeatures = 3, n_trees = 50, partial_sampling=0.7, max_depth = 4)

# Train the model on the dataset 
DecisionTree.fit!(model, x_train, y_train)

# Apply the trained model to the test features data set 
prediction = convert(Array{Int64,1}, DecisionTree.predict(model, x_test))


for i in 1:size(inputDB, 1)
  if (cocamidesOrNot(i) == true)
      tempRow = dfExtract(i, names(inputDB)[3:end])
      push!(tempRow, inputDBcocamide[findRowNumber(i)[end:end], "predictRi"][1])
      push!(dfOnlyCocamides, tempRow)
  else
      tempRow = dfExtract(i, names(inputDB)[3:end])
      push!(tempRow, Float64(0))
      push!(dfWithoutCocamides, tempRow)
  end
end

# df with 30684 x 2+15977+1
dfOnlyCocamides
# ouputing df 28302 x (2+15977)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamides.csv"
CSV.write(savePath, dfOnlyCocamides)

# df with 28302 or less x 2+15977+1
dfWithoutCocamides
# ouputing df 28302 x (2+15977)
savePath = "D:\\0_data\\dataframeCNLsRows_dfWithoutCocamides.csv"
CSV.write(savePath, dfWithoutCocamides)