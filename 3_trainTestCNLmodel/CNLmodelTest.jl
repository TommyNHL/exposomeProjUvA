VERSION
using Pkg
Pkg.add("StatsPlots")
Pkg.add("Plots")
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
using StatsPlots
using Plots
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
using ScikitLearn.GridSearch: GridSearchCV

# inputing 4862 x (3+15994+1)
# columns: SMILES, INCHIKEY, CNLs, predictRi
inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOnlyCocamides.csv", DataFrame)
#inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOutsideCocamides.csv", DataFrame)
#inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOutsideCocamidesInDA.csv", DataFrame)

# 15998 -> 15994 columns
## data for model testing
x_test = deepcopy(inputDB)
select!(x_test, Not([:ENTRY, :SMILES, :INCHIKEY, :FPpredictRi]));
size(x_test)
y_test = deepcopy(inputDB[:, end])
size(y_test)

#load a model
# requires python 3.11 or 3.12
modelRF_CNL = jl.load("D:\\1_model\\CocamideExtended_CNLsRi.joblib")
size(modelRF_CNL)

# performace
## Maximum absolute error
## mean square error (MSE) calculation
## Root mean square error (RMSE) calculation
function errorDetermination(arrRi, predictedRi)
    sumAE = 0
    maxAE = 0
    for i = 1:size(predictedRi, 1)
        AE = abs(arrRi[i] - predictedRi[i])
        if (AE > maxAE)
            maxAE = AE
        end
        sumAE += (AE ^ 2)
    end
    MSE = sumAE / size(predictedRi, 1)
    RMSE = MSE ^ 0.5
    return maxAE, MSE, RMSE
end

## R-square value
function rSquareDetermination(arrRi, predictedRi)
    sumY = 0
    for i = 1:size(predictedRi, 1)
        sumY += predictedRi[i]
    end
    meanY = sumY / size(predictedRi, 1)
    sumAE = 0
    sumRE = 0
    for i = 1:size(predictedRi, 1)
        AE = abs(arrRi[i] - predictedRi[i])
        RE = abs(arrRi[i] - meanY)
        sumAE += (AE ^ 2)
        sumRE += (RE ^ 2)
    end
    rSquare = 1 - (sumAE / sumRE)
    return rSquare
end

## Average accuracy
function avgAcc(arrAcc, cv)
    sumAcc = 0
    for acc in arrAcc
        sumAcc += acc
    end
    return sumAcc / cv
end

# external validation, CNL-predictedRi vs. FP-predictedRi
# requires sklearn v1.3.1 has been installed on Python 3.11 environment
predictedRi_test = predict(modelRF_CNL, Matrix(x_test))
inputDB[!, "CNLpredictRi"] = predictedRi_test
# save, ouputing trainSet df 0.7 x (3+15994+1)
savePath = "D:\\0_data\\dataframeCNLsRows_dfNonCocamidesTest_withCNLPredictedRi.csv"
CSV.write(savePath, inputDB)

maxAE_test, MSE_test, RMSE_test = errorDetermination(y_test, predictedRi_test)
rSquare_test = rSquareDetermination(y_test, predictedRi_test)
## accuracy
acc1_test = score(modelRF_CNL, Matrix(x_test), Vector(y_test))
acc5_test = cross_val_score(modelRF_CNL, Matrix(x_test), Vector(y_test); cv = 5)
avgAcc_test = avgAcc(acc5_test, 5)

# plots
plotTest = marginalkde(
        y_test, 
        predictedRi_test, 
        xlabel = "FP-derived Ri values", 
        ylabel = "CNL-derived Ri values", 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300
        )
plot!(plotTest.spmap[:contour], 
        y_test -> y_test, c=:red, 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
        # Saving
savefig(plotTest, "D:\\2_output\\CNLRiPrediction_Test.png")