VERSION
using Pkg
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()

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
#pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

#input csv of Ref, 15 columns df, including
    # Name, Formula, Mass, CAS, ChemSpider, IUPAC, 
    # SubMix, Label, M+H, Frags, Int, INCHIKEY, Name_MB, simple_name, FragsAll
    # Name- 
    inputRef = CSV.read("J:\\UvA\\CNL_Ref.csv", DataFrame)
    describe(inputRef)[7, :]
    describe(inputRef)[8, :]
    inputRef[:, "SubMix"]
    inputRef[:, "Label"]

    inputRef[!, "PestMixOrNot"] .= Integer(0)

#extract PesticideMix 1 - 8
    function check1to8(str)
        if ((length(str) >= 9) && (str[1:14] == "PesticideMix 1"))
            return 1
        elseif ((length(str) >= 9) && (str[1:14] == "PesticideMix 2"))
            return 2
        elseif ((length(str) >= 9) && (str[1:14] == "PesticideMix 3"))
            return 3
        elseif ((length(str) >= 9) && (str[1:14] == "PesticideMix 4"))
            return 4
        elseif ((length(str) >= 9) && (str[1:14] == "PesticideMix 5"))
            return 5
        elseif ((length(str) >= 9) && (str[1:14] == "PesticideMix 6"))
            return 6
        elseif ((length(str) >= 9) && (str[1:14] == "PesticideMix 7"))
            return 7
        elseif ((length(str) >= 9) && (str[1:14] == "PesticideMix 8"))
            return 8
        else
            return 0
        end
    end

    # check1to8
    for i in 1:size(inputRef, 1)
        println(i)
        result = check1to8(inputRef[i, "SubMix"])
        if (result != 0)
            inputRef[i, "PestMixOrNot"] = result
        else
            inputRef[i, "PestMixOrNot"] = 0
        end
    end

    inputRef[:, "PestMixOrNot"]

    inputRef = inputRef[inputRef.PestMixOrNot .!= 0, :]
    sort!(inputRef, [:PestMixOrNot, :INCHIKEY])

    inputRef[:, "INCHIKEY"]

#save csv
    savePath = "F:\\CNL_Ref_PestMix_1-8.csv"
    CSV.write(savePath, inputRef)

#gather distinct INCHIKEY IDs
    distinctKeys = Set()
    for i in 1:size(inputRef, 1)
        key = inputRef[i, "INCHIKEY"]
        if (key !== missing)
            push!(distinctKeys, key)
        end
    end
    distinctKeys = sort!(collect(distinctKeys))

#export distinct INCHIKEYs
    outputDf = DataFrame([distinctKeys], ["PesticideMixINCHIKEYs"])

    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_1-8.csv"
    CSV.write(savePath, outputDf)