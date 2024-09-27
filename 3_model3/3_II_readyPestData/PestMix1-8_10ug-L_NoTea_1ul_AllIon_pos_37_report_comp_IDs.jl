#download
    #INCHIKEYs_CNL_Ref_PestMix_1-8.csv
    #dataAllFP_withNewPredictedRiWithStratification.csv
    #TPTN_dfCNLfeaturesStr.csv
    #CocamideExtended73_CNLsRi_RFwithStratification.joblib
    #modelTPTNModeling_6paraKNN_noFilterWithDeltaRI.joblib

#import packages
    VERSION
    using Pkg
    #Pkg.add("ScikitLearn")
    import Conda
    Conda.PYTHONDIR
    ENV["PYTHON"] = raw"C:\Users\T1208\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
    Pkg.build("PyCall")
    Pkg.status()
    #Pkg.add(PackageSpec(url=""))
    using Random
    using CSV, DataFrames, Conda, LinearAlgebra, Statistics
    using PyCall
    using StatsPlots
    using Plots
    using ProgressBars
    
    jl = pyimport("joblib")             # used for loading models
    f1_score = pyimport("sklearn.metrics").f1_score
    matthews_corrcoef = pyimport("sklearn.metrics").matthews_corrcoef
    make_scorer = pyimport("sklearn.metrics").make_scorer
    f1 = make_scorer(f1_score, pos_label=1, average="binary")
    
    using ScikitLearn  #: @sk_import, fit!, predict
    @sk_import ensemble: RandomForestRegressor
    @sk_import ensemble: GradientBoostingClassifier
    @sk_import linear_model: LogisticRegression
    @sk_import ensemble: RandomForestClassifier
    @sk_import ensemble: AdaBoostClassifier
    @sk_import tree: DecisionTreeClassifier
    @sk_import metrics: recall_score
    @sk_import neural_network: MLPClassifier
    @sk_import svm: SVC
    @sk_import neighbors: KNeighborsClassifier
    @sk_import inspection: permutation_importance
    #using ScikitLearn.GridSearch: RandomizedSearchCV
    using ScikitLearn.CrossValidation: cross_val_score
    using ScikitLearn.CrossValidation: train_test_split
    #using ScikitLearn.GridSearch: GridSearchCV

#import groud truth
    INCHIKEYreal = Array(CSV.read("F:\\UvA\\INCHIKEYs_CNL_Ref_PestMix_1-8.csv", DataFrame)[:,1])

#handle MS/MS data
    # inputing __ x 20 dfs -> 29914 x 12+4+1 df
    ## ID, Rt, MS1Mass, Name, Formula, ACCESSION, 
    ## RefMatchFrag, UsrMatchFrag, MS1Error, MS2Error, MS2ErrorStd, 
    ## DirectMatch, ReversMatch, Probability, FinalScore, 
    ## SpecType, MatchedFrags, Inchikey, FragMZ, FragInt
    inputDB1 = CSV.read("F:\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs.csv", DataFrame)
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
    
    savePath = "F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_check.csv"
    CSV.write(savePath, combinedDB)
    
    outputDf = combinedDB[:, ["INCHIKEY_ID", "Inchikey", "ID", "RefMatchFragRatio", "UsrMatchFragRatio", 
    "MS1Error", "MS2Error", "MS2ErrorStd", "DirectMatch", "ReversMatch", 
    "FinalScoreRatio", "MS1Mass", "FragMZ", "LABEL"]]
    
    # output csv is a __ x 14 df
    savePath = "F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_checked.csv"
    CSV.write(savePath, outputDf)

    #handle the delta Ri feature
        # creating a 68120 x 2+8+2+1+1 df, 
            ## columns: INCHIKEY_ID, INCHIKEYreal, 8+1 ULSA features, LABEL
            ##                      FP->Ri, CNL->Ri ^
        # matching INCHIKEY, 30684 x 793 df
        inputFP2Ri = CSV.read("F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
        sort!(inputFP2Ri, [:INCHIKEY, :SMILES])

        # FP-derived Ri values
        outputDf[!, "FPpredictRi"] .= float(0)

        for i in 1:size(outputDf, 1)
            println(i)
            if outputDf[i, "Inchikey"] in Array(inputFP2Ri[:, "INCHIKEY"])
                rowNo = findall(inputFP2Ri.INCHIKEY .== outputDf[i, "Inchikey"])
                outputDf[i, "FPpredictRi"] = inputFP2Ri[rowNo[end:end], "predictRi"][1]
            else
                outputDf[i, "FPpredictRi"] = float(8888888)
            end
        end
        # filtering in INCHIKEY_ID with Ri values
        outputDf = outputDf[outputDf.FPpredictRi .!= float(8888888), :]
        sort!(outputDf, [:LABEL, :INCHIKEY_ID])

#output csv __ x 15 df
    savePath = "F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_ready4CNLdf.csv"
    CSV.write(savePath, outputDf)

    inputTPTNdf = CSV.read("F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_ready4CNLdf.csv", DataFrame)

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

    # initialization for 1 more column -> __ x 15+1
    inputTPTNdf[!, "CNLmasses"] .= [[]]
    size(inputTPTNdf)

    # import pre-defined 15961 CNL features
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
    for i in 1:size(inputTPTNdf, 1)
        println(i)
        fragIons = getVec(inputTPTNdf[i,"FragMZ"])
        arrNL = Set()
        for frag in fragIons
            if (inputTPTNdf[i,"MS1Mass"] - frag) >= float(0)
                NL = round((inputTPTNdf[i,"MS1Mass"] - frag), digits = 2)
                if (NL in candidatesList)
                    push!(arrNL, NL)
                end
            end
        end
        inputTPTNdf[i, "CNLmasses"] = sort!(collect(arrNL))
    end
    sort!(inputTPTNdf, [:LABEL, :INCHIKEY_ID, :CNLmasses])
    
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
    for i in 1:size(inputTPTNdf, 1)
        println(i)
        arr = []
        arr = getMasses(inputTPTNdf, i, arr, "dig")#"str")
        if (size(arr, 1) >= 2)
            push!(retain, i)
        end
    end
    inputTPTNdf = inputTPTNdf[retain, :]
    savePath = "F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_extractedWithCNLsList.csv"
    CSV.write(savePath, inputTPTNdf)

    inputTPTNdf = CSV.read("F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_extractedWithCNLsList.csv", DataFrame)

    # storing data in a Matrix
    X = zeros(3453, 15961)

    for i in 1:3453
        println(i)
        arr = []
        arr = getMasses(inputTPTNdf, i, arr, "str")#"dig")
        mumIon = round(inputTPTNdf[i, "MS1Mass"], digits = 2)
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
        insertcols!(dfCNLs, 1, ("ENTRY"=>collect(1:3453)))
        insertcols!(dfCNLs, 2, ("INCHIKEY_ID"=>inputTPTNdf[:, "INCHIKEY_ID"]))
        insertcols!(dfCNLs, 3, ("INCHIKEY"=>inputTPTNdf[:, "Inchikey"]))
        insertcols!(dfCNLs, 4, ("INCHIKEYreal"=>inputTPTNdf[:, "ID"]))
        insertcols!(dfCNLs, 5, ("RefMatchFragRatio"=>inputTPTNdf[:, "RefMatchFragRatio"]))
        insertcols!(dfCNLs, 6, ("UsrMatchFragRatio"=>inputTPTNdf[:, "UsrMatchFragRatio"]))
        insertcols!(dfCNLs, 7, ("MS1Error"=>inputTPTNdf[:, "MS1Error"]))
        insertcols!(dfCNLs, 8, ("MS2Error"=>inputTPTNdf[:, "MS2Error"]))
        insertcols!(dfCNLs, 9, ("MS2ErrorStd"=>inputTPTNdf[:, "MS2ErrorStd"]))
        insertcols!(dfCNLs, 10, ("DirectMatch"=>inputTPTNdf[:, "DirectMatch"]))
        insertcols!(dfCNLs, 11, ("ReversMatch"=>inputTPTNdf[:, "ReversMatch"]))
        insertcols!(dfCNLs, 12, ("FinalScoreRatio"=>inputTPTNdf[:, "FinalScoreRatio"]))
        insertcols!(dfCNLs, 13, ("MONOISOTOPICMASS"=>((inputTPTNdf[:, "MS1Mass"] .- 1.007276)/1000)))
    dfCNLs[!, "FPpredictRi"] = inputTPTNdf[:,"FPpredictRi"]
    size(dfCNLs)  # __ x (13+15961+1)

    # checking
    desStat = describe(dfCNLs)[12:end-1, :]
    
    #load a pre-trained CNL-to-Ri model
        # requires python 3.11 or 3.12
        modelRF_CNL = jl.load("F:\\UvA\\CocamideExtended73_CNLsRi_RFwithStratification.joblib")
        size(modelRF_CNL)
    
    # predict CNL-derived Ri
    CNLpredictedRi = predict(modelRF_CNL, Matrix(dfCNLs[:, 13:end-1]))
    dfCNLs[!, "CNLpredictRi"] = CNLpredictedRi
    dfCNLs[!, "DeltaRi"] = (CNLpredictedRi - dfCNLs[:, "FPpredictRi"]) / 1000
    dfCNLs[!, "LABEL"] = inputTPTNdf[:, "LABEL"]
        
    # saving csv
    savePath = "F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_withCNLRideltaRi.csv"
        CSV.write(savePath, dfCNLs)
        println("done for saving csv")

    dfOutput = CSV.read("F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_withCNLRideltaRi.csv", DataFrame)

#leverage
    describe(dfOutput)[end-4:end-2, :]
    X = deepcopy(dfOutput[:, 13:end-4])  # __ x 15962 df
    size(X)
    Y = deepcopy(dfOutput[:, end-2])
    size(Y)
    Xmat = Matrix(X)
    
    # 15962 x 15962
    hipinv = zeros(15962, 15962)
    hipinv[:,:] .= pinv(Xmat'*Xmat)
    
    function leverage_dist(X)
        h = zeros(3453,1)
        for i in ProgressBar(1: size(X,1)) #check dimensions
            x = X[i,:]
            #hi = x'*pinv(X'*X)*x
            hi = x'* hipinv *x
            #push!(h,hi)
            h[i,1] = hi
        end
        return h
    end
    
    h = leverage_dist(Matrix(X))
    ht = Vector(transpose(h)[1,:])
    
    dfOutput[!, "Leverage"] .= ht
    dfOutput = dfOutput[:, vcat(collect(1:13), end-4, end-3, end-2, end-1, end)]

    # saving csv
    savePath = "F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_dataframeTPTNModeling.csv"
    CSV.write(savePath, dfOutput)

#TP/TN prediction
    inputDB_test = CSV.read("F:\\UvA\\F\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
    sort!(inputDB_test, [:ENTRY])
    insertcols!(inputDB_test, 10, ("MatchDiff"=>float(0)))
    inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
    inputDB_test = inputDB_test[inputDB_test.Leverage .<= 0.14604417882015916, :]
    #describe(inputDB_test[inputDB_test.LABEL .== 0, :])
    #describe(inputDB_test[inputDB_test.LABEL .== 1, :])
    #inputDB_test = inputDB_test[inputDB_test.MS1Error .>= float(-0.061), :]
    #inputDB_test = inputDB_test[inputDB_test.MS1Error .<= float(0.058), :]
    for i = 1:size(inputDB_test, 1)
        inputDB_test[i, "RefMatchFragRatio"] = log10(inputDB_test[i, "RefMatchFragRatio"])
        inputDB_test[i, "UsrMatchFragRatio"] = log10(inputDB_test[i, "UsrMatchFragRatio"])
        inputDB_test[i, "FinalScoreRatio"] = log10(inputDB_test[i, "FinalScoreRatio"])
        inputDB_test[i, "MatchDiff"] = inputDB_test[i, "DirectMatch"] - inputDB_test[i, "ReversMatch"]
        inputDB_test[i, "MONOISOTOPICMASS"] = log10(inputDB_test[i, "MONOISOTOPICMASS"])
        if inputDB_test[i, "DeltaRi"] !== float(0)
            inputDB_test[i, "DeltaRi"] = inputDB_test[i, "DeltaRi"] * float(-1)
        end
    end
    describe(inputDB_test[:, 5:14])
    for f = 5:14
        avg = float(mean(inputDB_test[:, f]))
        top = float(maximum(inputDB_test[:, f]))
        down = float(minimum(inputDB_test[:, f]))
        for i = 1:size(inputDB_test, 1)
            inputDB_test[i, f] = (inputDB_test[i, f] - avg) / (top - down)
        end
    end
    # save, ouputing 583 x 18+1 df, 0:269; 1:314 = 
    savePath = "F:\\UvA\\F\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_dataframeTPTNModeling_0d5FinalScoreRatioDEFilterSTD.csv"
    CSV.write(savePath, inputDB_test)
    inputDB_test[inputDB_test.LABEL .== 1, :]

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
        function avgScore(arrAcc, cv)
            sumAcc = 0
            for acc in arrAcc
                sumAcc += acc
            end
            return sumAcc / cv
        end

        describe((inputDB_test))[vcat(5, 7,9, 13,14, 17), :]
        # load the pre-trained TP/TN model
        # requires python 3.11 or 3.12
        model = jl.load("F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_6paraKNN_noFilterWithDeltaRI.joblib")

        # predict TP/TN
        predictedTPTN_test = predict(model, Matrix(inputDB_test[:, vcat(5, 7,9, 13,14, 17)]))
        inputDB_test[!, "withDeltaRIpredictTPTN"] = predictedTPTN_test
        # save, ouputing testSet df 583 x 19+1 df
        savePath = "F:\\UvA\\F\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_withDeltaRIandPredictedTPTN_KNN.csv"
        CSV.write(savePath, inputDB_test)

    #show prediction performance
        inputTestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\F\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_withDeltaRIandPredictedTPTN_KNN.csv", DataFrame)
        describe((inputTestDB_withDeltaRiTPTN))[end-5:end, :]

        # 1, 0.3361921097770154, 0.5798207565938075
        maxAE_val, MSE_val, RMSE_val = errorDetermination(inputTestDB_withDeltaRiTPTN[:, end-2], inputTestDB_withDeltaRiTPTN[:, end])
        # -0.2575440758919678
        rSquare_val = rSquareDetermination(inputTestDB_withDeltaRiTPTN[:, end-2], inputTestDB_withDeltaRiTPTN[:, end])

        # 583 × 2 Matrix
        pTP_test = predict_proba(model, Matrix(inputTestDB_withDeltaRiTPTN[:, vcat(5, 7,9, 13,14, 17)]))
        # 0.5605095541401274
        recall_test = recall_score(Vector(inputTestDB_withDeltaRiTPTN[:, end-2]), predict(model, Matrix(inputTestDB_withDeltaRiTPTN[:, vcat(5, 7,9, 13,14, 17)])))

        inputTestDB_withDeltaRiTPTN[!, "p(0)"] = pTP_test[:, 1]
        inputTestDB_withDeltaRiTPTN[!, "p(1)"] = pTP_test[:, 2]
        # save, ouputing trainSet df 583 x 19+1+2 df
        savePath = "F:\\UvA\\F\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_withDeltaRIandPredictedTPTNandpTP_KNN.csv"
        CSV.write(savePath, inputTestDB_withDeltaRiTPTN)

        describe((inputTestDB_withDeltaRiTPTN))[end-4:end, :]

    #count individual sample performance
        inputTestDB_withDeltaRiTPTN = CSV.read("F:\\UvA\\F\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
        sort!(inputTestDB_withDeltaRiTPTN, [:INCHIKEYreal, :INCHIKEY])

        function checkID(inID, refID)
            if (inID == refID)
                return true
            else
                return false
            end
        end

        function countID(count, inID, refID)
            acc = count
            if (checkID(inID, refID) == true)
                acc += 1
                return acc
            else
                return 1
            end
        end

        function countP(pro, accPro, inID, refID)
            acc = accPro
            if (checkID(inID, refID) == true)
                acc += pro
                return acc
            else
                return pro
            end
        end

        count = 1
        colCount = [1]
        accP0 = inputTestDB_withDeltaRiTPTN[1, "p(0)"]
        colP0 = [accP0]
        accP1 = inputTestDB_withDeltaRiTPTN[1, "p(1)"]
        colP1 = [accP1]
        colIDSummary = []
        for id in 1:size(inputTestDB_withDeltaRiTPTN[:, "INCHIKEY"], 1) - 1
            count = countID(count, inputTestDB_withDeltaRiTPTN[id+1, "INCHIKEY"], inputTestDB_withDeltaRiTPTN[id, "INCHIKEY"])
            accP0 = countP(inputTestDB_withDeltaRiTPTN[id+1, "p(0)"], accP0, inputTestDB_withDeltaRiTPTN[id+1, "INCHIKEY"], inputTestDB_withDeltaRiTPTN[id, "INCHIKEY"])
            accP1 = countP(inputTestDB_withDeltaRiTPTN[id+1, "p(1)"], accP1, inputTestDB_withDeltaRiTPTN[id+1, "INCHIKEY"], inputTestDB_withDeltaRiTPTN[id, "INCHIKEY"])
            push!(colCount, count)
            push!(colP0, accP0)
            push!(colP1, accP1)
            if (colCount[id] >= colCount[id+1])
                push!(colIDSummary, 1)
            else
                push!(colIDSummary, 0)
            end
        end
        push!(colIDSummary, 1)

        inputTestDB_withDeltaRiTPTN[!, "countID"] = colCount
        inputTestDB_withDeltaRiTPTN[!, "countP(0)"] = colP0 ./ colCount
        inputTestDB_withDeltaRiTPTN[!, "countP(1)"] = colP1 ./ colCount
        inputTestDB_withDeltaRiTPTN[!, "colIDSummary"] = colIDSummary

        inputTestDB_withDeltaRiTPTN = inputTestDB_withDeltaRiTPTN[inputTestDB_withDeltaRiTPTN."colIDSummary" .== 1, :]

        savePath = "F:\\UvA\\F\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_withDeltaRIandPredictedTPTNandpTP_KNN_ind.csv"
        CSV.write(savePath, inputTestDB_withDeltaRiTPTN)
