#download
    #INCHIKEYs_CNL_Ref_PestMix_1-8.csv
    #dataAllFP_withNewPredictedRiWithStratification.csv
    #TPTN_dfCNLfeaturesStr.csv
    #CocamideExtended_CNLsRi_RFwithStratification.joblib
    #modelTPTNModeling_withDeltaRi.joblib

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
f1_score = pyimport("sklearn.metrics").f1_score
matthews_corrcoef = pyimport("sklearn.metrics").matthews_corrcoef

using ScikitLearn
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
@sk_import metrics: recall_score
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

#import groud truth
    INCHIKEYreal = Array(CSV.read("F:\\INCHIKEYs_CNL_Ref_PestMix_1-8.csv", DataFrame)[:,1])

#handle MS/MS data
    # inputing 29914 x 20 dfs -> 29914 x 12+4+1 df
    ## ID, Rt, MS1Mass, Name, Formula, ACCESSION, 
    ## RefMatchFrag, UsrMatchFrag, MS1Error, MS2Error, MS2ErrorStd, 
    ## DirectMatch, ReversMatch, Probability, FinalScore, 
    ## SpecType, MatchedFrags, Inchikey, FragMZ, FragInt
    inputDB1 = CSV.read("F:\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs.csv", DataFrame)
    #inputDB5 = CSV.read("F:\\Cand_synth_rr10_4001_5000.csv", DataFrame)
    #combinedDB = vcat(inputDB1, inputDB2, inputDB3, inputDB4, inputDB5)
    combinedDB = inputDB1[:, ["ID", 
        #"INCHIKEYreal", "INCHIKEY", 
        "Inchikey", 
        "RefMatchFrag", "UsrMatchFrag", 
        "MS1Error", "MS2Error", "MS2ErrorStd", 
        "DirectMatch", "ReversMatch", 
        "FinalScore", "MS1Mass", "FragMZ"]]
    
    combinedDB = combinedDB[combinedDB.RefMatchFrag .!= "NaN", :]
    combinedDB = combinedDB[combinedDB.RefMatchFrag .!== NaN, :]
    combinedDB = combinedDB[combinedDB.UsrMatchFrag .!= "NaN", :]
    combinedDB = combinedDB[combinedDB.UsrMatchFrag .!== NaN, :]
    combinedDB = combinedDB[combinedDB.MS1Error .!= "NaN", :]
    combinedDB = combinedDB[combinedDB.MS1Error .!== NaN, :]
    combinedDB = combinedDB[combinedDB.MS2Error .!= "NaN", :]
    combinedDB = combinedDB[combinedDB.MS2ErrorStd .!== NaN, :]
    combinedDB = combinedDB[combinedDB.DirectMatch .!= "NaN", :]
    combinedDB = combinedDB[combinedDB.DirectMatch .!== NaN, :]
    combinedDB = combinedDB[combinedDB.ReversMatch .!= "NaN", :]
    combinedDB = combinedDB[combinedDB.ReversMatch .!== NaN, :]
    combinedDB = combinedDB[combinedDB.FinalScore .!= "NaN", :]
    combinedDB = combinedDB[combinedDB.FinalScore .!== NaN, :]
    combinedDB = combinedDB[combinedDB.Inchikey .!= "N/A", :]
    combinedDB[!, "RefMatchFragRatio"] .= float(0)
    combinedDB[!, "UsrMatchFragRatio"] .= float(0)
    combinedDB[!, "FinalScoreRatio"] .= float(0)
    combinedDB[!, "INCHIKEY_ID"] .= ""

    savePath = "F:\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_check.csv"
    CSV.write(savePath, combinedDB)
    
    #handle 8 features
    function takeRatio(str)
        num = ""
        ratio = float(0)
        for i in 1:length(str)
            if num == "-"
                num = ""
                continue
            elseif str[i] == '-'
                ratio = parse(Float64, num)
                #print(ratio)
                num = "-"
            elseif str[i] != '-'
                num = string(num, str[i])
                #print(num)
            end
        end
        ratio = ratio / (parse(Float64, num))
        return ratio
    end

    ratioRef = []
    ratioUsr = []
    ratioScore = []
    trueOrFalse = []
    for i in 1:size(combinedDB, 1)
        println(i)
        push!(ratioRef, takeRatio(combinedDB[i, "RefMatchFrag"]))
        push!(ratioUsr, takeRatio(combinedDB[i, "UsrMatchFrag"]))
        push!(ratioScore, combinedDB[i, "FinalScore"]/7)
        combinedDB[i, "INCHIKEY_ID"] = string(combinedDB[i, "Inchikey"], "_", string(combinedDB[i, "ID"]))
        if (String31(combinedDB[i, "Inchikey"]) in INCHIKEYreal)
            push!(trueOrFalse, 1)
        else
            push!(trueOrFalse, 0)
        end
    end

    combinedDB[!, "RefMatchFragRatio"] = ratioRef
    combinedDB[!, "UsrMatchFragRatio"] = ratioUsr
    combinedDB[!, "FinalScoreRatio"] = ratioScore
    combinedDB[!, "LABEL"] = trueOrFalse

    #handle the delta Ri feature
        # creating a 68120 x 2+8+2+1+1 df, 
            ## columns: INCHIKEY_ID, INCHIKEYreal, 8+1 ULSA features, LABEL
            ##                      FP->Ri, CNL->Ri ^
        # matching INCHIKEY, 30684 x 793 df
        inputFP2Ri = CSV.read("F:\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
        sort!(inputFP2Ri, [:INCHIKEY, :SMILES])

        # FP-derived Ri values
        combinedDB[!, "FPpredictRi"] .= float(0)

        for i in 1:size(combinedDB, 1)
            println(i)
            if combinedDB[i, "Inchikey"] in Array(inputFP2Ri[:, "INCHIKEY"])
                rowNo = findall(inputFP2Ri.INCHIKEY .== combinedDB[i, "Inchikey"])
                combinedDB[i, "FPpredictRi"] = inputFP2Ri[rowNo[end:end], "predictRi"][1]
            else
                combinedDB[i, "FPpredictRi"] = float(8888888)
            end
        end

        outputDf = combinedDB[combinedDB.FPpredictRi .!= float(8888888), :]
        sort!(outputDf, [:LABEL, :INCHIKEY_ID])

# output csv is a 68120 x 2+8+2+1+1 df
    outputDf = outputDf[:, ["INCHIKEY_ID", 
        #"INCHIKEY", "INCHIKEYreal", 
        "Inchikey", 
        "RefMatchFragRatio", "UsrMatchFragRatio", "MS1Error", "MS2Error", 
        "MS2ErrorStd", "DirectMatch", "ReversMatch", "FinalScoreRatio", 
        "MS1Mass", "FragMZ", 
        "FPpredictRi", "LABEL"]]
    
    savePath = "F:\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_ready4CNLdf.csv"
    CSV.write(savePath, outputDf)

    outputDf = CSV.read("F:\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_ready4CNLdf.csv", DataFrame)

#create CNL df
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

    # initialization for 1 more column -> 68120 x 14+1
    outputDf[!, "CNLmasses"] .= [[]]

    # import pre-defined CNL features
    CNLfeaturesStr = CSV.read("F:\\TPTN_dfCNLfeaturesStr.csv", DataFrame)[:, "CNLfeaturesStr"]

        candidatesList = []
        for can in CNLfeaturesStr
            #push!(candidatesList, round(parse(Float64, can), digits = 2))
            push!(candidatesList, round(can, digits = 2))
        end
        CNLfeaturesStr = []
        for can in candidatesList
            #push!(candidatesList, round(parse(Float64, can), digits = 2))
            push!(CNLfeaturesStr, string(can))
        end

    # CNL extraction
    for i in 1:size(outputDf, 1)
        println(i)
        fragIons = getVec(outputDf[i,"FragMZ"])
        arrNL = Set()
        for frag in fragIons
            if (outputDf[i,"MS1Mass"] - frag) >= float(0)
                NL = round((outputDf[i,"MS1Mass"] - frag), digits = 2)
                if (NL in candidatesList)
                    push!(arrNL, NL)
                end
            end
        end
        outputDf[i, "CNLmasses"] = sort!(collect(arrNL))
    end
    sort!(outputDf, [:LABEL, :INCHIKEY_ID, :CNLmasses])
    
    # Reducing df size (rows)
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
    for i in 1:size(outputDf,1)
        if (size(outputDf[i, "CNLmasses"], 1) >= 2)
            push!(retain, i)
        end
    end
    outputDf = outputDf[retain, :]

    # storing data in a Matrix
    X = zeros(17385, 15961)

    for i in 1:17385
        println(i)
        arr = []
        arr = getMasses(outputDf, i, arr, "dig")
        mumIon = round(outputDf[i, "MS1Mass"], digits = 2)
        for col in arr
            mz = findall(x->x==col, candidatesList)
            if (col <= mumIon)
                X[i, mz] .= 1
            elseif (col > mumIon)
                X[i, mz] .= -1
            end
        end
    end

    # creating df with 11 + 1(monoisotopic mass) + 15961(CNLs)
    dfCNLs = DataFrame(X, CNLfeaturesStr)
        insertcols!(dfCNLs, 1, ("ENTRY"=>collect(1:17385)))
        insertcols!(dfCNLs, 2, ("INCHIKEY_ID"=>outputDf[:, "INCHIKEY_ID"]))
        insertcols!(dfCNLs, 3, ("INCHIKEY"=>outputDf[:, "Inchikey"]))
        insertcols!(dfCNLs, 4, ("RefMatchFragRatio"=>outputDf[:, "RefMatchFragRatio"]))
        insertcols!(dfCNLs, 5, ("UsrMatchFragRatio"=>outputDf[:, "UsrMatchFragRatio"]))
        insertcols!(dfCNLs, 6, ("MS1Error"=>outputDf[:, "MS1Error"]))
        insertcols!(dfCNLs, 7, ("MS2Error"=>outputDf[:, "MS2Error"]))
        insertcols!(dfCNLs, 8, ("MS2ErrorStd"=>outputDf[:, "MS2ErrorStd"]))
        insertcols!(dfCNLs, 9, ("DirectMatch"=>outputDf[:, "DirectMatch"]))
        insertcols!(dfCNLs, 10, ("ReversMatch"=>outputDf[:, "ReversMatch"]))
        insertcols!(dfCNLs, 11, ("FinalScoreRatio"=>outputDf[:, "FinalScoreRatio"]))
        insertcols!(dfCNLs, 12, ("ISOTOPICMASS"=>outputDf[:, "MS1Mass"] .- 1.007276))
    dfCNLs[!, "FPpredictRi"] = outputDf[:,"FPpredictRi"]

    # checking
    desStat = describe(dfCNLs)[12:end-1, :]
    
    #load a pre-trained CNL-to-Ri model
        # requires python 3.11 or 3.12
        modelRF_CNL = jl.load("F:\\CocamideExtended_CNLsRi_RFwithStratification.joblib")
        size(modelRF_CNL)
    
    # predict CNL-derived Ri
    CNLpredictedRi = predict(modelRF_CNL, Matrix(dfCNLs[:, 12:end-1]))
    dfCNLs[!, "CNLpredictRi"] = CNLpredictedRi
    dfCNLs[!, "DeltaRi"] = (CNLpredictedRi - dfCNLs[:, "FPpredictRi"]) / 1000
    dfCNLs[!, "LABEL"] = outputDf[:, "LABEL"]
        
    # saving csv
    savePath = "F:\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_withCNLRi.csv"
        CSV.write(savePath, dfCNLs)
        println("done for saving csv")

    dfCNLs = CSV.read("F:\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_withCNLRi.csv", DataFrame)

#TP/TN prediction
    inputDB_test = deepcopy(dfCNLs)
    sort!(inputDB_test, [:ENTRY])

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

    # load the pre-trained TP/TN model
        # requires python 3.11 or 3.12
        modelRF_TPTN = jl.load("F:\\modelTPTNModeling_WholeWithDeltaRi.joblib")
        size(modelRF_TPTN)

        # predict TP/TN
        describe(inputDB_test)[1:5, :]
        describe(inputDB_test)[6:10, :]
        describe(inputDB_test)[11:15, :]
        describe(inputDB_test)[end-4:end, :]
        predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(4:11), end-1)]))
        1 in inputDB_test[:, "LABEL"]
        1 in predictedTPTN_test
        inputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_test
        # save, ouputing testSet df 18183 x 15978 df
        savePath = "F:\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_TPTN.csv"
        CSV.write(savePath, inputDB_test)

#show prediction performance
    # 1.0, 0.014161353306223489, 0.11900148447067158
    maxAE_val, MSE_val, RMSE_val = errorDetermination(Matrix(inputDB_test[:, vcat(collect(4:11), end-2)]), predictedTPTN_test)
    # -0.22812402319094782
    rSquare_val = rSquareDetermination(Matrix(inputDB_test[:, vcat(collect(4:11), end-2)]), predictedTPTN_test)
    ## accuracy, 0.8211516251443656
    acc1_val = score(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(4:11), end-2)]), Vector(inputDB_test[:, end-1]))
    # 0.7909585876918, 0.8006929549579277, 0.8091074080184788
    acc5_val = cross_val_score(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(4:11), end-2)]), Vector(inputDB_test[:, end-1]); cv = 3)
    # 0.8002529835560689
    avgAcc_val = avgAcc(acc5_val, 3)
    # 18183 × 2 Matrix
    pTP_test = predict_proba(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(4:11), end-2)]))
    # 0.046334310850439875
    f1_test = f1_score(Vector(inputDB_test[:, end-1]), predictedTPTN_test)
    # 0.12186712050169676
    mcc_test = matthews_corrcoef(Vector(inputDB_test[:, end-1]), predictedTPTN_test)

    inputDB_test[!, "Prob(0)"] = pTP_test[:, 1]
    inputDB_test[!, "Prob(1)"] = pTP_test[:, 2]
    
    describe(inputDB_test)[end-7:end, :]
    
    # save, ouputing 18333 x 19 df
    outputCSV = inputDB_test[:, vcat(collect(1:12), collect(end-6:end))]
    savePath = "F:\\PestMix1-8_1000ug-L_NoTea_1ul_AllIon_pos_52_report_comp_IDs_withPredictedTPTNandpTP.csv"
    CSV.write(savePath, outputCSV)
    