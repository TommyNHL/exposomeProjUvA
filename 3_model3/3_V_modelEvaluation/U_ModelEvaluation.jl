## INPUT(S)
# trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv
# testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv
# noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv
# TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv

## OUTPUT(S)
# modelTPTNModeling_6paraKNN_noFilterWithDeltaRI.joblib
# modelTPTNModeling_6paraKNN_noFilterWithOutDeltaRI.joblib
# dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTN_KNN.csv
# dataframeTPTNModeling_TrainDF_withOutDeltaRIandPredictedTPTN_KNN.csv
# dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_TrainDF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_ValDF_withDeltaRIandPredictedTPTN_KNN.csv
# dataframeTPTNModeling_ValDF_withOutDeltaRIandPredictedTPTN_KNN.csv
# dataframeTPTNModeling_ValDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_ValDF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_PestDF_withDeltaRIandPredictedTPTN_KNN.csv
# dataframeTPTNModeling_PestDF_withOutDeltaRIandPredictedTPTN_KNN.csv
# dataframeTPTNModeling_PestDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_PestDF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_Pest2DF_withDeltaRIandPredictedTPTN_KNN.csv
# dataframeTPTNModeling_Pest2DF_withOutDeltaRIandPredictedTPTN_KNN.csv
# dataframeTPTNModeling_Pest2DF_withDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_Pest2DF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv

VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
import Conda
Conda.PYTHONDIR
#ENV["PYTHON"] = raw"C:\Users\T1208\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
using Random
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots
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

## import packages from Python ##
jl = pyimport("joblib")             # used for loading models
f1_score = pyimport("sklearn.metrics").f1_score
matthews_corrcoef = pyimport("sklearn.metrics").matthews_corrcoef
make_scorer = pyimport("sklearn.metrics").make_scorer
f1 = make_scorer(f1_score, pos_label=1, average="binary")

## input training set ##, 1686319 x 22 df
# 0: 1535009; 1: 151310 = 0.5493; 5.5724
trainDEFSDf = CSV.read("G:\\Temp\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
trainDEFSDf[trainDEFSDf.LABEL .== 1, :]
    ## calculate weight ##
    Yy_train = deepcopy(trainDEFSDf[:, end-4])  # 0.5493; 5.5724
    sampleW = []
    for w in Vector(Yy_train)
        if w == 0
            push!(sampleW, 0.5493)
        elseif w == 1
            push!(sampleW, 5.5724)
        end
    end

## input testing set ## 421381 x 22 df
# 0: 383416; 1: 37965 = 0.5495; 5.5496
testDEFSDf = CSV.read("G:\\Temp\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
testDEFSDf[testDEFSDf.LABEL .== 1, :]
    ## calculate weight ##
    Yy_val = deepcopy(testDEFSDf[:, end-4])  # 0.5495; 5.5496
    sampletestW = []
    for w in Vector(Yy_val)
        if w == 0
            push!(sampletestW, 0.5495)
        elseif w == 1
            push!(sampletestW, 5.5496)
        end
    end

## reconstruct a whole set ## spike blank (No Tea)
# 2107700 x 22 df; 
# 1686319+421381= 2107700, 0:1918425; 1:189275 = 
wholeDEFSDf = vcat(trainDEFSDf, testDEFSDf)
sort!(wholeDEFSDf, [:ENTRY])
wholeDEFSDf[wholeDEFSDf.LABEL .== 1, :]

## input validation set ##
# 10908 x 19 df
# 0: 7173; 1: 3735 = 0.7604; 1.4602
noTeaDEFSDf = CSV.read("G:\\Temp\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
noTeaDEFSDf[noTeaDEFSDf.LABEL .== 1, :]
    ## calculate weight ##
    Yy_test = deepcopy(noTeaDEFSDf[:, end-1])  # 0.7604; 1.4602
    samplepestW = []
    for w in Vector(Yy_test)
        if w == 0
            push!(samplepestW, 0.7604)
        elseif w == 1
            push!(samplepestW, 1.4602)
        end
    end

## input real sample set ## with Tea
# 29599 x 19 df
# 1: 8187
TeaDEFSDf = CSV.read("G:\\Temp\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)
TeaDEFSDf = TeaDEFSDf[TeaDEFSDf.LABEL .== 1, :]
    ## calculate weight ##
    Yy_test2 = deepcopy(TeaDEFSDf[:, end-1])
    samplepest2W = []
    for w in Vector(Yy_test2)
        if w == 0
            push!(samplepest2W, 0)
        elseif w == 1
            push!(samplepest2W, 0.5)
        end
    end

## define functions for performace evaluation ##
    # Maximum absolute error
    # mean square error (MSE) calculation
    # Root mean square error (RMSE) calculation
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
    #
    # Average score
    function avgScore(arrAcc, cv)
        sumAcc = 0
        for acc in arrAcc
            sumAcc += acc
        end
        return sumAcc / cv
    end


# ==================================================================================================
## train a k-Nearest Neighbors model ##, with DeltaRi
model = KNeighborsClassifier(
      n_neighbors = 379, 
      weights = "uniform", 
      leaf_size = 300, 
      p = 2, 
      metric = "minkowski"
      )
    #
    ## select features ##
    rank = vcat(5, 7,9, 13,14, 17)
    #
    ## balance samples ##
    N_train = trainDEFSDf
    M_train = vcat(trainDEFSDf, trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
        trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
        trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
        trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
        trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
        trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
        trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
        trainDEFSDf[trainDEFSDf.LABEL .== 1, :], 
        trainDEFSDf[trainDEFSDf.LABEL .== 1, :])
    #
    ## ready materials ##
    M_val = testDEFSDf
    M_pest = noTeaDEFSDf
    M_pest2 = TeaDEFSDf
    Xx_train = deepcopy(M_train[:, rank])
    nn_train = deepcopy(N_train[:, rank])
    Xx_val = deepcopy(M_val[:, rank])
    Xx_test = deepcopy(M_pest[:, rank])
    Xx_test2 = deepcopy(M_pest2[:, rank])
    Yy_train = deepcopy(M_train[:, end-4])
    mm_train = deepcopy(N_train[:, end-4])
    Yy_val = deepcopy(M_val[:, end-4])
    Yy_test = deepcopy(M_pest[:, end-1])
    Yy_test2 = deepcopy(M_pest2[:, end-1])
    #
    ## fit ##
    fit!(model, Matrix(Xx_train), Vector(Yy_train))
    importances = permutation_importance(model, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
    print(importances["importances_mean"])
        #"importances_std"  => [0.00365488, 0.0042455, 0.000954662, 0.00135283, 0.00220259, 0.00230624]
        #"importances_mean" => [0.00671984, 0.107105, 0.00196186, 0.000385039, 0.0118812, -0.00462963]
## save model ##
modelSavePath = "F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_6paraKNN_noFilterWithDeltaRI.joblib"
jl.dump(model, modelSavePath, compress = 5)


# ==================================================================================================
## train a k-Nearest Neighbors model without retention index error ##
model_noRI = KNeighborsClassifier(
      n_neighbors = 379, 
      weights = "uniform", 
      leaf_size = 300, 
      p = 2, 
      metric = "minkowski"
      )
    #
    ## select features ##
    rank2 = vcat(5, 7,9, 13,14)
    #
    ## ready materials ##
    Xx_train_noRI = deepcopy(M_train[:, rank2])
    nn_train_noRI = deepcopy(N_train[:, rank2])
    Xx_val_noRI = deepcopy(M_val[:, rank2])
    Xx_test_noRI = deepcopy(M_pest[:, rank2])
    Xx_test2_noRI = deepcopy(M_pest2[:, rank2])
    #
    ## fit ##
    fit!(model_noRI, Matrix(Xx_train_noRI), Vector(Yy_train))
    importances2 = permutation_importance(model_noRI, Matrix(Xx_test_noRI), Vector(Yy_test), n_repeats=10, random_state=42)
    print(importances2["importances_mean"])
        #"importances_std"  => [0.0038459, 0.00517923, 0.0017152, 0.00203555, 0.00159498, 0.00266296]
        #"importances_mean" => [0.0126421, 0.0797488, 0.00553722, -0.00176934, 0.00484965, 0.00163183]
## save model ##
modelSavePath = "F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_6paraKNN_noFilterWithOutDeltaRI.joblib"
jl.dump(model_noRI, modelSavePath, compress = 5)


# ==================================================================================================
## deploy models for training set ##
    ## load a model ##, with DeltaRi
    # requires python 3.11 or 3.12
    #model = jl.load("F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_6paraKNN_noFilterWithDeltaRI.joblib")
    model = jl.load("G:\\Temp\\modelTPTNModeling_6paraKNN_noFilterWithDeltaRI.joblib")
        #
        ## deploy model ##
        predictedTPTN_train = predict(model, Matrix(trainDEFSDf[:, rank]))
        train_withRI = deepcopy(trainDEFSDf)
        train_withRI[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
        #
    ## save ##, ouputing trainSet df 1686319 x 23 df
    savePath = "G:\\Temp\\dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTN_KNN.csv"
    CSV.write(savePath, train_withRI)
    #
    ## load a model ##, without DeltaRi
    # requires python 3.11 or 3.12
    model_noRI = jl.load("F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_6paraKNN_noFilterWithOutDeltaRI.joblib")
        #
        ## deploy model ##
        predictedTPTN_train = predict(model_noRI, Matrix(trainDEFSDf[:, rank2]))
        train_withoutRI = deepcopy(trainDEFSDf)
        train_withoutRI[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
        #
    ## save ##, ouputing trainSet df 1686319 x 23 df
    savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_TrainDF_withOutDeltaRIandPredictedTPTN_KNN.csv"
    CSV.write(savePath, train_withoutRI)


# ==================================================================================================
## evaluate predictive performance of the model with DeltaRi ##
inputDB_withDeltaRiTPTN = CSV.read("G:\\Temp\\dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTN_KNN.csv", DataFrame)
    #
    maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])  # 1, 0.21323664146581994, 0.46177553147153644
    pTP_train = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, rank]))  # 1686319 × 2 Matrix
    f1_train = f1_score(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end], sample_weight=sampleW)  # 0.8895717113929443
    mcc_train = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end], sample_weight=sampleW)  # 0.7738684612060831
    inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
    inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
    #
    ## save ##, ouputing trainSet df 1686319 x (23+2)
    savePath = "G:\\Temp\\dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv"
    CSV.write(savePath, inputDB_withDeltaRiTPTN)

## evaluate predictive performance of the model without DeltaRi ##
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_TrainDF_withOutDeltaRIandPredictedTPTN_KNN.csv", DataFrame)
    #
    maxAE_train, MSE_train, RMSE_train = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])  # 1, 0.2081682054225802, 0.45625454016653927
    rSquare_train = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])  # -0.6695318478337433
    pTP_train = predict_proba(model_noRI, Matrix(inputDB_withoutDeltaRiTPTN[:, rank2]))  # 1686319 × 2 Matrix
    f1_train = f1_score(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end], sample_weight=sampleW)  # 0.895403648988781
    mcc_train = matthews_corrcoef(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end], sample_weight=sampleW)  # 0.7873822419345989
    inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_train[:, 1]
    inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_train[:, 2]
    #
    ## save ##, ouputing trainSet df 1686319 x (23+2)
    savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_TrainDF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv"
    CSV.write(savePath, inputDB_withoutDeltaRiTPTN)


# ==================================================================================================
## deploy models for testing set ##
    ## load a model ##, with DeltaRi
    # requires python 3.11 or 3.12
    model = jl.load("G:\\Temp\\modelTPTNModeling_6paraKNN_noFilterWithDeltaRI.joblib")
        #
        ## deploy model ##
        predictedTPTN_val = predict(model, Matrix(testDEFSDf[:, rank]))
        val_withRI = deepcopy(testDEFSDf)
        val_withRI[!, "withDeltaRipredictTPTN"] = predictedTPTN_val
        #
    ## save ##, ouputing valSet df 421381 x 23 df
    savePath = "G:\\Temp\\dataframeTPTNModeling_ValDF_withDeltaRIandPredictedTPTN_KNN.csv"
    CSV.write(savePath, val_withRI)
    #
    ## load a model ##, without DeltaRi
    # requires python 3.11 or 3.12
    model_noRI = jl.load("F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_6paraKNN_noFilterWithOutDeltaRI.joblib")
        #
        ## deploy model ##
        predictedTPTN_val = predict(model_noRI, Matrix(testDEFSDf[:, rank2]))
        val_withoutRI = deepcopy(testDEFSDf)
        val_withoutRI[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_val
        #
    ## save ##, ouputing valSet df 421381 x 23 df
    savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_ValDF_withOutDeltaRIandPredictedTPTN_KNN.csv"
    CSV.write(savePath, val_withoutRI)


# ==================================================================================================
## evaluate predictive performance of the model with DeltaRi ##
inputDB_withDeltaRiTPTN = CSV.read("G:\\Temp\\dataframeTPTNModeling_ValDF_withDeltaRIandPredictedTPTN_KNN.csv", DataFrame)
    #
    maxAE_val, MSE_val, RMSE_val = errorDetermination(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end])  # 1, 0.2153965176408049, 0.4641083037835079
    pTP_val = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, rank]))  # 421381 × 2 Matrix
    f1_val = f1_score(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end], sample_weight=sampletestW)  # 0.8874084658561004
    mcc_val = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-5], inputDB_withDeltaRiTPTN[:, end], sample_weight=sampletestW)  # 0.7689142728578838
    inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_val[:, 1]
    inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_val[:, 2]
    ## save ##, ouputing valSet df 421381 x (23+2)
    savePath = "G:\\Temp\\dataframeTPTNModeling_ValDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv"
    CSV.write(savePath, inputDB_withDeltaRiTPTN)

## evaluate predictive performance of the model without DeltaRi ##
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_ValDF_withOutDeltaRIandPredictedTPTN_KNN.csv", DataFrame)
    #
    maxAE_val, MSE_val, RMSE_val = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])  # 1, 0.20958704830070649, 0.45780678053159773
    rSquare_val = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end])  # -0.669871035967158
    pTP_val = predict_proba(model_noRI, Matrix(inputDB_withoutDeltaRiTPTN[:, rank2]))  # 421381 × 2 Matrix
    f1_val = f1_score(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end], sample_weight=sampletestW)  # 0.8943354518762837
    mcc_val = matthews_corrcoef(inputDB_withoutDeltaRiTPTN[:, end-5], inputDB_withoutDeltaRiTPTN[:, end], sample_weight=sampletestW)  # 0.784990521464426
    inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_val[:, 1]
    inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_val[:, 2]
    ## save ##, ouputing valSet df 421381 x (23+2)
    savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_ValDF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv"
    CSV.write(savePath, inputDB_withoutDeltaRiTPTN)


# ==================================================================================================
## deploy models for validation set ## (NoTea Pest, spike blank)
    ## load a model ##, with DeltaRi
    # requires python 3.11 or 3.12
    model = jl.load("G:\\Temp\\modelTPTNModeling_6paraKNN_noFilterWithDeltaRI.joblib")
        #
        ## deploy model ##
        predictedTPTN_pest = predict(model, Matrix(noTeaDEFSDf[:, rank]))
        pest_withRI = deepcopy(noTeaDEFSDf)
        pest_withRI[!, "withDeltaRipredictTPTN"] = predictedTPTN_pest
        #
    ## save ##, ouputing pestSet df 10908 x 20 df
    savePath = "G:\\Temp\\dataframeTPTNModeling_PestDF_withDeltaRIandPredictedTPTN_KNN.csv"
    CSV.write(savePath, pest_withRI)
    #
    ## load a model ##, without DeltaRi
    # requires python 3.11 or 3.12
    model_noRI = jl.load("F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_6paraKNN_noFilterWithOutDeltaRI.joblib")
        #
        ## deploy model ##
        predictedTPTN_pest = predict(model_noRI, Matrix(noTeaDEFSDf[:, rank2]))
        pest_withoutRI = deepcopy(noTeaDEFSDf)
        pest_withoutRI[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_pest
    ## save ##, ouputing pestSet df 10908 x 20 df
    savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_PestDF_withOutDeltaRIandPredictedTPTN_KNN.csv"
    CSV.write(savePath, pest_withoutRI)


# ==================================================================================================
## evaluate predictive performance of the model with DeltaRi ##
inputDB_withDeltaRiTPTN = CSV.read("G:\\Temp\\dataframeTPTNModeling_PestDF_withDeltaRIandPredictedTPTN_KNN.csv", DataFrame)
    ##
    maxAE_pest, MSE_pest, RMSE_pest = errorDetermination(inputDB_withDeltaRiTPTN[:, end-2], inputDB_withDeltaRiTPTN[:, end])  # 1, 0.3534103410341034, 0.594483255469911
    pTP_pest = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, rank]))  # 10908 × 2 Matrix
    f1_pest = f1_score(inputDB_withDeltaRiTPTN[:, end-2], inputDB_withDeltaRiTPTN[:, end], sample_weight=samplepestW)  # 0.6526465712267591
    mcc_pest = matthews_corrcoef(inputDB_withDeltaRiTPTN[:, end-2], inputDB_withDeltaRiTPTN[:, end], sample_weight=samplepestW)  # 0.2990024830957098
    inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_pest[:, 1]
    inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_pest[:, 2]
    ## save ##, ouputing pestSet df 10908 x (20+2)
    savePath = "G:\\Temp\\dataframeTPTNModeling_PestDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv"
    CSV.write(savePath, inputDB_withDeltaRiTPTN)

## evaluate predictive performance of the model without DeltaRi ##
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_PestDF_withOutDeltaRIandPredictedTPTN_KNN.csv", DataFrame)
    #
    maxAE_pest, MSE_pest, RMSE_pest = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-2], inputDB_withoutDeltaRiTPTN[:, end])  # 1, 0.3502933626696003, 0.5918558630862757
    rSquare_pest = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-2], inputDB_withoutDeltaRiTPTN[:, end])  # -0.4680324802812619
    pTP_pest = predict_proba(model_noRI, Matrix(inputDB_withoutDeltaRiTPTN[:, rank2]))  # 10908 × 2 Matrix
    f1_pest = f1_score(inputDB_withoutDeltaRiTPTN[:, end-2], inputDB_withoutDeltaRiTPTN[:, end], sample_weight=samplepestW)  # 0.6537818399417835
    mcc_pest = matthews_corrcoef(inputDB_withoutDeltaRiTPTN[:, end-2], inputDB_withoutDeltaRiTPTN[:, end], sample_weight=samplepestW)  # 0.3033305164298841
    inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_pest[:, 1]
    inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_pest[:, 2]
    ## save ##, ouputing pestSet df 10908 x (20+2)
    savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_PestDF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv"
    CSV.write(savePath, inputDB_withoutDeltaRiTPTN)


# ==================================================================================================
## deploy models for real sample set ## (withTea Pest)
    ## load a model ##, with DeltaRi
    # requires python 3.11 or 3.12
    model = jl.load("G:\\Temp\\modelTPTNModeling_6paraKNN_noFilterWithDeltaRI.joblib")
        #
        ## deploy model ##
        predictedTPTN_pest2 = predict(model, Matrix(TeaDEFSDf[:, rank]))
        pest2_withRI = deepcopy(TeaDEFSDf)
        pest2_withRI[!, "withDeltaRipredictTPTN"] = predictedTPTN_pest2
        #
    ## save ##, ouputing pest2Set df 8187 x 20 df
    savePath = "G:\\Temp\\dataframeTPTNModeling_Pest2DF_withDeltaRIandPredictedTPTN_KNN.csv"
    CSV.write(savePath, pest2_withRI)
    #
    ## load a model ##, without DeltaRi
    # requires python 3.11 or 3.12
    model_noRI = jl.load("F:\\UvA\\F\\UvA\\app\\modelTPTNModeling_6paraKNN_noFilterWithOutDeltaRI.joblib")
        #
        ## deploy model ##
        predictedTPTN_pest2 = predict(model_noRI, Matrix(TeaDEFSDf[:, rank2]))
        pest2_withoutRI = deepcopy(TeaDEFSDf)
        pest2_withoutRI[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_pest2
    ## save ##, ouputing pest2Set df 8187 x 20 df
    savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_Pest2DF_withOutDeltaRIandPredictedTPTN_KNN.csv"
    CSV.write(savePath, pest2_withoutRI)


# ==================================================================================================
## evaluate predictive performance of the model with DeltaRi ##
inputDB_withDeltaRiTPTN = CSV.read("G:\\Temp\\dataframeTPTNModeling_Pest2DF_withDeltaRIandPredictedTPTN_KNN.csv", DataFrame)
    #
    maxAE_pest2, MSE_pest2, RMSE_pest2 = errorDetermination(inputDB_withDeltaRiTPTN[:, end-2], inputDB_withDeltaRiTPTN[:, end])  # 1, 0.40356663002320753, 0.6352689430652245
    pTP_pest2 = predict_proba(model, Matrix(inputDB_withDeltaRiTPTN[:, rank]))  # 8187 × 2 Matrix
    recall_pest2 = recall_score(Vector(inputDB_withDeltaRiTPTN[:, end-2]), predict(model, Matrix(inputDB_withDeltaRiTPTN[:, rank])))  # 0.5964333699767925
    recall_pest2_ = recall_score(Vector(inputDB_withDeltaRiTPTN[:, end-2]), predict(model, Matrix(inputDB_withDeltaRiTPTN[:, rank])), sample_weight=samplepest2W)  # 0.5964333699767925
    inputDB_withDeltaRiTPTN[!, "p(0)"] = pTP_pest2[:, 1]
    inputDB_withDeltaRiTPTN[!, "p(1)"] = pTP_pest2[:, 2]
    ## save ##, ouputing pest2Set df 8187 x (20+2)
    savePath = "G:\\Temp\\dataframeTPTNModeling_Pest2DF_withDeltaRIandPredictedTPTNandpTP_KNN.csv"
    CSV.write(savePath, inputDB_withDeltaRiTPTN)

## evaluate predictive performance of the model without DeltaRi ##
inputDB_withoutDeltaRiTPTN = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_Pest2DF_withOutDeltaRIandPredictedTPTN_KNN.csv", DataFrame)
    #
    maxAE_pest2, MSE_pest2, RMSE_pest2 = errorDetermination(inputDB_withoutDeltaRiTPTN[:, end-2], inputDB_withoutDeltaRiTPTN[:, end])  # 1, 0.4570660803713204, 0.6760666242104548
    rSquare_pest2 = rSquareDetermination(inputDB_withoutDeltaRiTPTN[:, end-2], inputDB_withoutDeltaRiTPTN[:, end])  # -1.1878674505609874
    pTP_pest2 = predict_proba(model_noRI, Matrix(inputDB_withoutDeltaRiTPTN[:, rank2]))  # 8187 × 2 Matrix
    recall_pest2 = recall_score(Vector(inputDB_withoutDeltaRiTPTN[:, end-2]), predict(model_noRI, Matrix(inputDB_withoutDeltaRiTPTN[:, rank2])))  # 0.5429339196286797
    inputDB_withoutDeltaRiTPTN[!, "p(0)"] = pTP_pest2[:, 1]
    inputDB_withoutDeltaRiTPTN[!, "p(1)"] = pTP_pest2[:, 2]
    ## save ##, ouputing pest2Set df 8187 x (20+2)
    savePath = "F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_Pest2DF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv"
    CSV.write(savePath, inputDB_withoutDeltaRiTPTN)
    