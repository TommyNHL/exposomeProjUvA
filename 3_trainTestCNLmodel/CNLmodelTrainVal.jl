VERSION
using Pkg
#Pkg.add("PyCall")
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
#Pkg.add("MLJ")
#Pkg.add("JLD")
#Pkg.add("HDF5")
#Pkg.add("PyCallJLD")
#Pkg.add(Pkg.PackageSpec(;name="ScikitLearn", version="1.3.1"))
#using JLD, HDF5, PyCallJLD
#Pkg.add(PackageSpec(url=""))
#Pkg.add("MLDataUtils")
#Pkg.add(PackageSpec(url=""))
using Random
using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
#using PyPlot
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
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.GridSearch: GridSearchCV

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

function partitionTrainVal(df, ratio = 0.7)
    noOfRow = nrow(df)
    idx = shuffle(1:noOfRow)
    train_idx = view(idx, 1:floor(Int, ratio*noOfRow))
    test_idx = view(idx, (floor(Int, ratio*noOfRow)+1):noOfRow)
    df[train_idx,:], df[test_idx,:]
end

# give 2+15979+1 = 15980 columns
trainSet, valSet = partitionTrainVal(inputDB, 0.7)  # 70% train 30% Val/Test
size(trainSet)
size(valSet)

# ouputing trainSet df 0.7 x (2+15977+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamidesTrain.csv"
CSV.write(savePath, trainSet)

# ouputing trainSet df 0.3 x (2+15977+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamidesVal.csv"
CSV.write(savePath, valSet)

# 15980 -> 15977 columns
## data for model training
x_train = deepcopy(trainSet)
select!(x_train, Not([:SMILES, :INCHIKEY, :predictRi]));
size(x_train)
y_train = deepcopy(trainSet[:, end])
size(y_train)
## data for internal model validation
x_val = deepcopy(valSet)
select!(x_val, Not([:SMILES, :INCHIKEY, :predictRi]));
size(x_val)
y_val = deepcopy(valSet[:, end])
size(y_val)

# modeling
model = RandomForestRegressor()
param_dist = Dict(
      "n_estimators" => 50:50:300, 
      "max_depth" => 2:2:10, 
      "min_samples_leaf" => 8:8:32, 
      "max_features" => [Int64(ceil(size(x_train,2)/3))], 
      "n_jobs" => [-1], 
      "oob_score" => [true], 
      "random_state" => [1]
      )
gridsearch = GridSearchCV(model, param_dist)
@time fit!(gridsearch, Matrix(x_train), Vector(y_train))
println("Best parameters: $(gridsearch.best_params_)")

model = RandomForestRegressor(
      n_estimators = 100, 
      max_depth = 10, 
      min_samples_leaf = 8, 
      max_features = Int64(ceil(size(x_train,2)/3)), 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 1
      )
fit!(model, Matrix(x_train), Vector(y_train))
cross_val_score(model, Matrix(x_train), Vector(y_train); cv = 5)

# model validation
predictedRi = predict(model, Matrix(x_val))
# CNL-predictedRi vs. FP-predictedRi

# performace
## error calculation
for i = 1:size(predictedRi, 1)
    println(predictedRi[i], ", ", y_val[i])
end

# saving model
modelSavePath = "D:\\1_model\\CocamideExtended_CNLsRi.joblib"
jl.dump(model, modelSavePath, compress = 5)