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
using CSV, DataFrames, PyCall, Conda, LinearAlgebra, Statistics
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
np = pyimport("numpy")

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.GridSearch: GridSearchCV

# iniputing dfOnlyCocamides with 28 x (3+15994+1)
dfOnlyCocamides = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOnlyCocamides.csv", DataFrame)

# inputing dfOutsideCocamides with 4862 x (3+15994+1)
dfOutsideCocamides = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOutsideCocamides.csv", DataFrame)

#load a model
# requires python 3.11 or 3.12
modelRF_CNL = jl.load("D:\\1_model\\CocamideExtended_CNLsRi.joblib")
size(modelRF_CNL)

# inputing 4862 x (3+15994+1)
# columns: SMILES, INCHIKEY, CNLs, predictRi
#inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOnlyCocamides.csv", DataFrame)
inputTrainVal = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOutsideCocamides.csv", DataFrame)

# inputing 4862 x (3+15994+1)
# columns: SMILES, INCHIKEY, CNLs, predictRi
inputTest = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOnlyCocamides.csv", DataFrame)
#inputDB = CSV.read("D:\\0_data\\dataframeCNLsRows_dfOutsideCocamides.csv", DataFrame)

# determining Leverage values
# 3+15994+1+1
dfOnlyCocamides[!, "LeverageValue"] .= Float64(0)
dfOutsideCocamides[!, "LeverageValue"] .= Float64(0)

###
function leverageCal(matX)
    hiis = []
    matX_t = transpose(matX)
    for i in 1:size(matX, 1)
        for j in 1:size(matX, 2)
            hii = (matX[i] ./ matX) * (matX[j] ./ matX_t)
            push!(hiis, hii)
        end
    end
    return hiis
end
###

levOnlyCocamides = leverageCal(Matrix(dfOnlyCocamides[:, 4:end-2]))
levOutsideCocamides = leverageCal(Matrix(dfOutsideCocamides[:, 4:end-2]))

diff = 0.001
bins = collect(0:diff:1)
counts = zeros(length(bins)-1)
for i = 1:length(bins)-1
    counts[i] = sum(bins[i] .<= levOnlyCocamides .< bins[i+1])
end
thresholdAD = bins[findfirst(cumsum(counts)./sum(counts) .> 0.95)-1]


### plots in SI
sn = "Cocamides"
FPRi_train = Matrix(inputTrainVal[:, end])
CNLRi_train = predict(modelRF_CNL, Matrix(inputTrainVal[:, 4:end-1]))
plotAD4CNL_train = scatter([600], [600], markershape = :star, c = :white, label = "Outside AD", margin = (5, :mm), dpi = 300)
scatter!(FPRi_train[levOnlyCocamides.<= thresholdAD], CNLRi_train[levOnlyCocamides.<= thresholdAD], xlabel = "FP-derived "*sn*" RI", ylabel = "Predicted CNL-derived "*sn*" RI", label = "Training set", c = 2)
scatter!(FPRi_train[levOnlyCocamides.> thresholdAD], CNLRi_train[levOnlyCocamides.> thresholdAD], xlabel = "FP-derived "*sn*" RI", markershape = :star, label = "", c = 2)
plot!(plotAD4CNL_train.spmap[:contour], 
        FPRi_train -> FPRi_train, c=:red, 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
        # Saving
savefig(plotAD4CNL_train, "D:\\2_output\\plotAD4CNL_train.png")

sn = "Non-Cocamides"
FPRi_test = Matrix(inputTest[:, end])
CNLRi_test = predict(modelRF_CNL, Matrix(inputTest[:, 4:end-1]))
plotAD4CNL_test = scatter([600], [600], markershape = :star, c = :white, label = "Outside AD", margin = (5, :mm), dpi = 300)
scatter!(FPRi_test[levOnlyCocamides.<= thresholdAD], CNLRi_test[levOnlyCocamides.<= thresholdAD], xlabel = "FP-derived "*sn*" RI", ylabel = "Predicted CNL-derived "*sn*" RI", label = "Testing set", c = 2)
scatter!(FPRi_test[levOnlyCocamides.> thresholdAD], CNLRi_test[levOnlyCocamides.> thresholdAD], xlabel = "FP-derived "*sn*" RI", markershape = :star, label = "", c = 2)
plot!(plotAD4CNL_test.spmap[:contour], 
        FPRi_test -> FPRi_test, c=:red, 
        label = false, 
        margin = (5, :mm), 
        size = (600,600), 
        dpi = 300)
        # Saving
savefig(plotAD4CNL_test, "D:\\2_output\\plotAD4CNL_test.png")

# outputing dfOnlyCocamides with 28 x (3+15994+1)
outputTrainVal = inputTrainVal[inputTrainVal.ION_MODE .== "POSITIVE", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
FPRi_train[levOnlyCocamides.<= thresholdAD]
savePath = "D:\\0_data\\dataframeCNLsRows_dfOnlyCocamidesInAD.csv"
CSV.write(savePath, dfOnlyCocamides)

# outputing dfOutsideCocamides with 4862 x (3+15994+1)
inputTest
FPRi_test[levOnlyCocamides.<= thresholdAD]
savePath = "D:\\0_data\\dataframeCNLsRows_dfOutsideCocamidesInAD.csv"
CSV.write(savePath, dfOutsideCocamides)