#download
    #TPTN_dfCNLfeaturesStr.csv
    #CocamideExtended_CNLsRi_RFwithStratification.joblib

    VERSION
    using Pkg
    #Pkg.add("PyCall")
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
    pd = pyimport("padelpy")
    jl = pyimport("joblib")
    
    using ScikitLearn
    @sk_import ensemble: RandomForestRegressor
    @sk_import ensemble: RandomForestClassifier
    #using ScikitLearn.GridSearch: RandomizedSearchCV
    using ScikitLearn.CrossValidation: cross_val_score
    using ScikitLearn.CrossValidation: train_test_split
    #using ScikitLearn.GridSearch: GridSearchCV
    
    #handle MS/MS data
        function getVec(matStr)
            if matStr[1] .== '['
                if contains(matStr, ", ")
                    str = split(matStr[2:end-1],", ")
                else
                    str = split(matStr[2:end-1]," ")
                end
            elseif matStr[1] .== 'A'
                if contains(matStr, ", ")
                    str = split(matStr[5:end-1],", ")
                else
                    str = split(matStr[5:end-1]," ")
                end
            elseif matStr[1] .== 'F'
                if matStr .== "Float64[]"
                    return []
                else
                    str = split(matStr[9:end-1],", ")
                end
            elseif matStr[1] .== 'I'
                if matStr .== "Int64[]"
                    return []
                else
                    str = split(matStr[7:end-1],", ")
                end
            else
                println("New vector start")
                println(matStr)
            end
            if length(str) .== 1 && cmp(str[1],"") .== 0
                return []
            else
                str = parse.(Float64, str)
                return str
            end
        end
    
        # inputing csv of MS/MS, 31 columns df
        inputAllMSDB = CSV.read("__MSMSdb__", DataFrame)
    
        # filtering out NEGATIVE ionization mode by "NEGATIVE"
        # filtering out NEGATIVE ionization mode by "N"
        # filtering in precusor ion with measured m/z
        AllMSData = inputAllMSDB[inputAllMSDB.ION_MODE .!= "NEGATIVE", 
            ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES", "ION_MODE"]]
        AllMSData = AllMSData[AllMSData.ION_MODE .!= "N", 
            ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
        AllMSData = AllMSData[AllMSData.PRECURSOR_ION .!== NaN, 
            ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
    
        # initialization for 1 more column
        AllMSData[!, "CNLmasses"] .= [[]]
        size(AllMSData)
    
        # NLs calculation, filtering CNL-in-interest, storing in Vector{Any}
            # filtering in CNLs features according to the pre-defined CNLs in CNLs_10mDa.csv
            # inputing 15961 candidates
            CNLfeaturesStr = CSV.read("F:\\TPTN_dfCNLfeaturesStr.csv", DataFrame)[:, "CNLfeaturesStr"]
    
            # creating floar array
            candidatesList = []
            for can in CNLfeaturesStr
                push!(candidatesList, round(can, digits = 2))
            end
    
            # creating str array
            CNLfeaturesStr = []
            for can in candidatesList
                push!(CNLfeaturesStr, string(can))
            end
    
            for i in 1:size(inputAllMSDB, 1)
                println(i)
                fragIons = getVec(inputAllMSDB[i,"FragMZ"])
                arrNL = Set()
                for frag in fragIons
                    if (inputAllMSDB[i,"MS1Mass"] - frag) >= float(0)
                        NL = round((inputAllMSDB[i,"MS1Mass"] - frag), digits = 2)
                        if (NL in candidatesList)
                            push!(arrNL, NL)
                        end
                    end
                end
                inputAllMSDB[i, "CNLmasses"] = sort!(collect(arrNL))
            end
            
            sort!(inputAllMSDB, [:LABEL, :INCHIKEY_ID, :CNLmasses])
            inputAllMSDB[:, "CNLmasses"]
            
        #Reduce df size (rows)
            function getMasses(db, i, arr, arrType = "dig")
                massesArr = arr
                if (arrType == "dig")
                    masses = db[i, "CNLmasses"]
                elseif (arrType == "str")
                    masses = getVec(db[i, "CNLmasses"])
                end
                for mass in masses
                    push!(massesArr, mass)
                end
                return massesArr
            end
    
        # removing rows that has Frag-ion of interest < 2 (optional)
            retain = []
            for i in 1:size(inputAllMSDB,1)
                if (size(inputAllMSDB[i, "CNLmasses"], 1) >= 2)
                    push!(retain, i)
                end
            end
            inputAllMSDB = inputAllMSDB[retain, :]
            
            
            # creating a table with 1(monoisotopic mass) + 15961(CNLs)
            # storing data in a Matrix
            X = zeros(__NoOdRows__, 15961)
            
            #for i in 1:1076799
            for i in 1:__NoOdRows__
                println(i)
                arr = []
                arr = getMasses(inputAllMSDB, i, arr, "dig")
                mumIon = round(inputAllMSDB[i, "MS1Mass"], digits = 2)
                for col in arr
                    mz = findall(x->x==col, candidatesList)
                    if (col <= mumIon)
                        X[i, mz] .= 1
                    elseif (col > mumIon)
                        X[i, mz] .= -1
                    end
                end
            end
            
            # creating df with 1(monoisotopic mass) + 15961(CNLs)
            dfCNLs = DataFrame(X, CNLfeaturesStr)
            insertcols!(dfCNLs, 1, ("ISOTOPICMASS"=>inputTPTNdf[1:__NoOdRows__, "MS1Mass"] .- 1.007276))
            
            # checking
            desStat = describe(dfCNLs)
            
        #load a pre-trained CNL-to-Ri model
            # requires python 3.11 or 3.12
            modelRF_CNL = jl.load("F:\\CocamideExtended_CNLsRi_RFwithStratification.joblib")
            size(modelRF_CNL)
            
        # predict CNL-derived Ri
        CNLpredictedRi = predict(modelRF_CNL, Matrix(dfCNLs[:, :]))
        dfCNLs[!, "CNLpredictRi"] = CNLpredictedRi
       
        # saving csv
        savePath = "__csv__.csv"
        CSV.write(savePath, dfCNLs)
        println("done for saving csv")
    