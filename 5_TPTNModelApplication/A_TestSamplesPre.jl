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
    inputRef = CSV.read("F:\\CNL_Ref.csv", DataFrame)
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
    function setDistinct(df)
        distinctKeys = Set()
        for i in 1:size(df, 1)
            key = df[i, "INCHIKEY"]
            if (key !== missing)
                push!(distinctKeys, key)
            end
        end
        distinctKeys = sort!(collect(distinctKeys))
        return distinctKeys
    end

    distinctKeys1to8 = setDistinct(inputRef)

    #export distinct INCHIKEYs by Standard mixture
    inputRef1 = inputRef[inputRef.PestMixOrNot .== 1, :]
    inputRef2 = inputRef[inputRef.PestMixOrNot .== 2, :]
    inputRef3 = inputRef[inputRef.PestMixOrNot .== 3, :]
    inputRef4 = inputRef[inputRef.PestMixOrNot .== 4, :]
    inputRef5 = inputRef[inputRef.PestMixOrNot .== 5, :]
    inputRef6 = inputRef[inputRef.PestMixOrNot .== 6, :]
    inputRef7 = inputRef[inputRef.PestMixOrNot .== 7, :]
    inputRef8 = inputRef[inputRef.PestMixOrNot .== 8, :]

    distinctKeys1 = setDistinct(inputRef1)
    distinctKeys2 = setDistinct(inputRef2)
    distinctKeys3 = setDistinct(inputRef3)
    distinctKeys4 = setDistinct(inputRef4)
    distinctKeys5 = setDistinct(inputRef5)
    distinctKeys6 = setDistinct(inputRef6)
    distinctKeys7 = setDistinct(inputRef7)
    distinctKeys8 = setDistinct(inputRef8)

#export distinct INCHIKEYs
    outputDf = DataFrame([distinctKeys1to8], ["PesticideMixINCHIKEYs"])
    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_1-8.csv"
    CSV.write(savePath, outputDf)

    outputDf1 = DataFrame([distinctKeys1], ["PesticideMixINCHIKEYs"])
    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_1.csv"
    CSV.write(savePath, outputDf1)

    outputDf2 = DataFrame([distinctKeys2], ["PesticideMixINCHIKEYs"])
    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_2.csv"
    CSV.write(savePath, outputDf2)

    outputDf3 = DataFrame([distinctKeys3], ["PesticideMixINCHIKEYs"])
    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_3.csv"
    CSV.write(savePath, outputDf3)

    outputDf4 = DataFrame([distinctKeys4], ["PesticideMixINCHIKEYs"])
    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_4.csv"
    CSV.write(savePath, outputDf4)

    outputDf5 = DataFrame([distinctKeys5], ["PesticideMixINCHIKEYs"])
    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_5.csv"
    CSV.write(savePath, outputDf5)

    outputDf6 = DataFrame([distinctKeys6], ["PesticideMixINCHIKEYs"])
    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_6.csv"
    CSV.write(savePath, outputDf6)

    outputDf7 = DataFrame([distinctKeys7], ["PesticideMixINCHIKEYs"])
    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_7.csv"
    CSV.write(savePath, outputDf7)

    outputDf8 = DataFrame([distinctKeys8], ["PesticideMixINCHIKEYs"])
    savePath = "F:\\INCHIKEYs_CNL_Ref_PestMix_8.csv"
    CSV.write(savePath, outputDf8)
