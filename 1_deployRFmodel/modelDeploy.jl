VERSION
using Pkg
Pkg.add("PyCall")
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"
Pkg.build("PyCall")
Pkg.status()
Pkg.add("JLD")
Pkg.add("HDF5")
Pkg.add("PyCallJLD")
Pkg.add(Pkg.PackageSpec(;name="ScikitLearn", version="1.3.1"))
using JLD, HDF5, PyCallJLD
Pkg.add(PackageSpec(url=""))
using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
Conda.add("padelpy")
## import packages ##
#using PyCall, Conda                 #using python packages
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models

using ScikitLearn: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier

#import csv
inputAll = CSV.read("D:\\0_data\\dataAllFingerprinter_4RiPredict.csv", DataFrame)
inputData = Matrix(inputAll[:, 3:end])

#load a model
modelRF = jl.load("D:\\1_model\\CocamideExtended.joblib")
size(modelRF)

#apply
predictedRi = predict(modelRF, inputData)
inputAll[!, "predictRi"] = predictedRi
#fit!(modelRF, inputData)

# save
savePath = "D:\\0_data\\dataAllFP_withPredictedRi.csv"
CSV.write(savePath, inputAll)