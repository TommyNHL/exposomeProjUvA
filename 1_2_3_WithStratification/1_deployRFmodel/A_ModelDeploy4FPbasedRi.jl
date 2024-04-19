VERSION
using Pkg
#Pkg.add("PyCall")
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
#Pkg.add(PackageSpec(url=""))
using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
#Conda.add("joblib")
jl = pyimport("joblib")             # used for loading models

using ScikitLearn: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier

#import csv
# input csv is a 30684 x (2+790) df, columns include 
        #SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs
inputAll = CSV.read("F:\\UvA\\dataAllFingerprinter_4RiPredict.csv", DataFrame)
inputData = Matrix(inputAll[:, 3:end])

#load a model, RandomForestRegressor(max_features=310, min_samples_leaf=8, n_estimators=200), size 200
# requires python 3.11 or 3.12
modelRF = jl.load("F:\\CocamideExtendedWithStratification.joblib")
size(modelRF)

#apply
# requires sklearn v1.3.1 has been installed on Python 3.11 environment
predictedRi = predict(modelRF, inputData)
inputAll[!, "predictRi"] = predictedRi

# save
# output csv is a 30684 x (2+791) df, columns include 
        #SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs, 
        #and newly added one (FP-derived predicted Ri)
inputAll
savePath = "F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification.csv"
CSV.write(savePath, inputAll)