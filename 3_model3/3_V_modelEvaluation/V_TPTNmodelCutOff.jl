## INPUT(S)
# dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_ValDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_PestDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv
# dataframeTPTNModeling_Pest2DF_withDeltaRIandPredictedTPTNandpTP_KNN.csv

## OUTPUT(S)
# dataframePostPredict_TrainALLWithDeltaRI_KNN.csv
# dataframePostPredict_TestALLWithDeltaRI_KNN.csv
# dataframePostPredict_PestNoTeaWithDeltaRI_KNN.csv
# dataframePostPredict_Pest2WithTeaWithDeltaRI_KNN.csv
# TPTNPrediction_KNNtrainTestCM.png
# TPTNPrediction_KNNpestPest2CM.png
# dataframePostPredict_TPRFNRFDR_newTrainALL_KNN.csv
# dataframePostPredict_TPRFNRFDR_newTestALL_KNN.csv
# dataframePostPredict_TPRFNRFDR_newPestNoTea_KNN.csv
# TPTNPrediction_P1threshold2TPRFNRFDR_newTrainALLylims_KNN.png
# dataframePostPredict10FDR_TrainALLWithDeltaRI_KNN.csv
# dataframePostPredict10FDR_TestALLWithDeltaRI_KNN.csv
# dataframePostPredict10FDR_PestNoTeaWithDeltaRI_KNN.csv
# dataframePostPredict10FDR_Pest2WithTeaWithDeltaRI_KNN.csv

VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\T1208\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
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

## input ## 1686319 x 25 df; 421381 x 25 df; 10908 x 22 df; 8187 x 22 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage, withDeltaRipredictTPTN, p0, p1
inputDB_TrainWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_TestWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_ValDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_PestDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_Pest2WithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_Pest2DF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)


# ==================================================================================================
## prepare to plot confusion matrix for training set ##
inputDB_TrainWithDeltaRi[!, "CM"] .= String("")
    inputDB_TrainWithDeltaRi_TP = 0
    inputDB_TrainWithDeltaRi_FP = 0
    inputDB_TrainWithDeltaRi_TN = 0
    inputDB_TrainWithDeltaRi_FN = 0
    for i in 1:size(inputDB_TrainWithDeltaRi , 1)
        if (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
            inputDB_TrainWithDeltaRi[i, "CM"] = "TP"
            inputDB_TrainWithDeltaRi_TP += 1
        elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
            inputDB_TrainWithDeltaRi[i, "CM"] = "FP"
            inputDB_TrainWithDeltaRi_FP += 1
        elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
            inputDB_TrainWithDeltaRi[i, "CM"] = "TN"
            inputDB_TrainWithDeltaRi_TN += 1
        elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
            inputDB_TrainWithDeltaRi[i, "CM"] = "FN"
            inputDB_TrainWithDeltaRi_FN += 1
        end
    end
    #
    CM_TrainWith = zeros(2, 2)
    CM_TrainWith[2, 1] = inputDB_TrainWithDeltaRi_TP  #149,466
    CM_TrainWith[2, 2] = inputDB_TrainWithDeltaRi_FP  #357,741
    CM_TrainWith[1, 2] = inputDB_TrainWithDeltaRi_TN  #1,177,268
    CM_TrainWith[1, 1] = inputDB_TrainWithDeltaRi_FN  #1,844

## save ##, ouputing df 1686319 x 25+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TrainALLWithDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_TrainWithDeltaRi)


# ==================================================================================================
## prepare to plot confusion matrix for testing set ##
inputDB_TestWithDeltaRi[!, "CM"] .= String("")
    inputDB_TestWithDeltaRi_TP = 0
    inputDB_TestWithDeltaRi_FP = 0
    inputDB_TestWithDeltaRi_TN = 0
    inputDB_TestWithDeltaRi_FN = 0
    for i in 1:size(inputDB_TestWithDeltaRi , 1)
        if (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
            inputDB_TestWithDeltaRi[i, "CM"] = "TP"
            inputDB_TestWithDeltaRi_TP += 1
        elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
            inputDB_TestWithDeltaRi[i, "CM"] = "FP"
            inputDB_TestWithDeltaRi_FP += 1
        elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
            inputDB_TestWithDeltaRi[i, "CM"] = "TN"
            inputDB_TestWithDeltaRi_TN += 1
        elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
            inputDB_TestWithDeltaRi[i, "CM"] = "FN"
            inputDB_TestWithDeltaRi_FN += 1
        end
    end
    #
    CM_TestWith = zeros(2, 2)
    CM_TestWith[2, 1] = inputDB_TestWithDeltaRi_TP  #37,405
    CM_TestWith[2, 2] = inputDB_TestWithDeltaRi_FP  #90,204
    CM_TestWith[1, 2] = inputDB_TestWithDeltaRi_TN  #293,212
    CM_TestWith[1, 1] = inputDB_TestWithDeltaRi_FN  #560

## save ##, ouputing df 421381 x 25+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TestALLWithDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_TestWithDeltaRi)


# ==================================================================================================
## prepare to plot confusion matrix for validation set ##, No Tea spike blank
inputDB_PestWithDeltaRi[!, "CM"] .= String("")
    inputDB_PestWithDeltaRi_TP = 0
    inputDB_PestWithDeltaRi_FP = 0
    inputDB_PestWithDeltaRi_TN = 0
    inputDB_PestWithDeltaRi_FN = 0
    for i in 1:size(inputDB_PestWithDeltaRi , 1)
        if (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
            inputDB_PestWithDeltaRi[i, "CM"] = "TP"
            inputDB_PestWithDeltaRi_TP += 1
        elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
            inputDB_PestWithDeltaRi[i, "CM"] = "FP"
            inputDB_PestWithDeltaRi_FP += 1
        elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
            inputDB_PestWithDeltaRi[i, "CM"] = "TN"
            inputDB_PestWithDeltaRi_TN += 1
        elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
            inputDB_PestWithDeltaRi[i, "CM"] = "FN"
            inputDB_PestWithDeltaRi_FN += 1
        end
    end
    #
    CM_PestWith = zeros(2, 2)
    CM_PestWith[2, 1] = inputDB_PestWithDeltaRi_TP  #2,460
    CM_PestWith[2, 2] = inputDB_PestWithDeltaRi_FP  #2,580
    CM_PestWith[1, 2] = inputDB_PestWithDeltaRi_TN  #4,593
    CM_PestWith[1, 1] = inputDB_PestWithDeltaRi_FN  #1,275

## save ##, ouputing df 10908 x 22+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_PestNoTeaWithDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)


# ==================================================================================================
## prepare to plot confusion matrix for real sample set ##, With Tea
inputDB_Pest2WithDeltaRi[!, "CM"] .= String("")
    inputDB_Pest2WithDeltaRi_TP = 0
    inputDB_Pest2WithDeltaRi_FP = 0
    inputDB_Pest2WithDeltaRi_TN = 0
    inputDB_Pest2WithDeltaRi_FN = 0
    for i in 1:size(inputDB_Pest2WithDeltaRi , 1)
        if (inputDB_Pest2WithDeltaRi[i, "LABEL"] == 1 && inputDB_Pest2WithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
            inputDB_Pest2WithDeltaRi[i, "CM"] = "TP"
            inputDB_Pest2WithDeltaRi_TP += 1
        elseif (inputDB_Pest2WithDeltaRi[i, "LABEL"] == 0 && inputDB_Pest2WithDeltaRi[i, "withDeltaRipredictTPTN"] == 1)
            inputDB_Pest2WithDeltaRi[i, "CM"] = "FP"
            inputDB_Pest2WithDeltaRi_FP += 1
        elseif (inputDB_Pest2WithDeltaRi[i, "LABEL"] == 0 && inputDB_Pest2WithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
            inputDB_Pest2WithDeltaRi[i, "CM"] = "TN"
            inputDB_Pest2WithDeltaRi_TN += 1
        elseif (inputDB_Pest2WithDeltaRi[i, "LABEL"] == 1 && inputDB_Pest2WithDeltaRi[i, "withDeltaRipredictTPTN"] == 0)
            inputDB_Pest2WithDeltaRi[i, "CM"] = "FN"
            inputDB_Pest2WithDeltaRi_FN += 1
        end
    end
    #
    CM_Pest2With = zeros(2, 2)
    CM_Pest2With[2, 1] = inputDB_Pest2WithDeltaRi_TP  #4,883
    CM_Pest2With[2, 2] = inputDB_Pest2WithDeltaRi_FP  #0
    CM_Pest2With[1, 2] = inputDB_Pest2WithDeltaRi_TN  #0
    CM_Pest2With[1, 1] = inputDB_Pest2WithDeltaRi_FN  #3,304

## save ##, ouputing df 8187 x 22+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_Pest2WithTeaWithDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_Pest2WithDeltaRi)


# ==================================================================================================
## plot confusion matrix for training & testing sets ##
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
TrainOutplotCM = plot(layout = layout, link = :both, 
        size = (2000, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_TrainWith, cmap = :viridis, cbar = :true, 
        clims = (0, 600000), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Training Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n149,466"], subplot = 1, font(color="white"))
        annotate!(["0"], ["1"], ["FP\n357,741"], subplot = 1)
        annotate!(["1"], ["0"], ["FN\n1,844"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n1,177,268"], subplot = 1)
heatmap!(["1", "0"], ["0", "1"], CM_TestWith, cmap = :viridis, cbar = :true, 
        clims = (0, 150000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Testing Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n37,405"], subplot = 2, font(color="white"))
        annotate!(["0"], ["1"], ["FP\n90,204"], subplot = 2)
        annotate!(["1"], ["0"], ["FN\n560"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n293,212"], subplot = 2)
savefig(TrainOutplotCM, "F:\\UvA\\F\\UvA\\app\\TPTNPrediction_KNNtrainTestCM.png")


# ==================================================================================================
## plot confusion matrix for validation (No Tea Spike Blank) & real sample sets ##
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
PestOutplotCM = plot(layout = layout, link = :both, 
        size = (2000, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_PestWith, cmap = :viridis, cbar = :true, 
        clims = (0, 4500), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Spiked Pesticides with No-Tea Matrix Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n2,460"], subplot = 1, font(color="white"))
        annotate!(["0"], ["1"], ["FP\n2,580"], subplot = 1)
        annotate!(["1"], ["0"], ["FN\n1,275"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n4,593"], subplot = 1)
heatmap!(["1", "0"], ["0", "1"], CM_Pest2With, cmap = :viridis, cbar = :true, 
        clims = (0, 9000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Spiked Pesticides with Tea Matrix Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n4,883"], subplot = 2, font(color="white"))
        annotate!(["0"], ["1"], ["FP\n0"], subplot = 2, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n3,304"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n0"], subplot = 2, font(color="white"))
savefig(PestOutplotCM, "F:\\UvA\\F\\UvA\\app\\TPTNPrediction_KNNpestPest2CM.png")


# ==================================================================================================
## prepare to plot P(TP)threshold-to-TPR curve ## training set
    ##  1686319 x 26 df
    inputDB_TrainWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TrainALLWithDeltaRI_KNN.csv", DataFrame)
        sort!(inputDB_TrainWithDeltaRi, [:"p(1)"], rev = true)
        for i in 1:size(inputDB_TrainWithDeltaRi, 1)
            inputDB_TrainWithDeltaRi[i, "p(1)"] = round(float(inputDB_TrainWithDeltaRi[i, "p(1)"]), digits = 2)
        end
        #
    # 421381 x 26 df
    inputDB_TestWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TestALLWithDeltaRI_KNN.csv", DataFrame)
        sort!(inputDB_TestWithDeltaRi, [:"p(1)"], rev = true)
        for i in 1:size(inputDB_TestWithDeltaRi, 1)
            inputDB_TestWithDeltaRi[i, "p(1)"] = round(float(inputDB_TestWithDeltaRi[i, "p(1)"]), digits = 2)
        end
        #
    # 10908 x 23 df
    inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_PestNoTeaWithDeltaRI_KNN.csv", DataFrame)
        sort!(inputDB_PestWithDeltaRi, [:"p(1)"], rev = true)
        for i in 1:size(inputDB_PestWithDeltaRi, 1)
            inputDB_PestWithDeltaRi[i, "p(1)"] = round(float(inputDB_PestWithDeltaRi[i, "p(1)"]), digits = 2)
        end
        #
    # 8187 x 23 df
    inputDB_Pest2WithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_Pest2WithTeaWithDeltaRI_KNN.csv", DataFrame)
        sort!(inputDB_Pest2WithDeltaRi, [:"p(1)"], rev = true)
        for i in 1:size(inputDB_Pest2WithDeltaRi, 1)
            inputDB_Pest2WithDeltaRi[i, "p(1)"] = round(float(inputDB_Pest2WithDeltaRi[i, "p(1)"]), digits = 2)
        end
    #
    ## define a function for Confusion Matrix ##
    function get1rateTrain(df, thd)
        TP = 0  # 
        FN = 0  # 
        TN = 0  # 
        FP = 0  # 
        #TP2 = 0
        #FN2 = 0
        #TN2 = 0
        #FP2 = 0
        for i in 1:size(df , 1)
            if (df[i, "LABEL"] == 1 && df[i, "p(1)"] >= thd)
                TP += (1 * 5.5724)
                #TP2 += 1
            elseif (df[i, "LABEL"] == 1 && df[i, "p(1)"] < thd)
                FN += (1 * 5.5724)
                #FN2 += 1
            elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] >= thd)
                FP += (1 * 0.5493)
                #FP2 += 1
            elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] < thd)
                TN += (1 * 0.5493)
                #TN2 += 1
            end
        end
        return (TP / (TP + FN)), (FN / (TP + FN)), (FP / (FP + TP)), (FP / (FP + TN)), (TN / (TN + FP)), (TP / (TP + FP)), (TP / (TP + FN))
    end
    #
    function get1rateTest(df, thd)
        TP = 0  # 
        FN = 0  # 
        TN = 0  # 
        FP = 0  # 
        #TP2 = 0
        #FN2 = 0
        #TN2 = 0
        #FP2 = 0
        for i in 1:size(df , 1)
            if (df[i, "LABEL"] == 1 && df[i, "p(1)"] >= thd)
                TP += (1 * 5.5496)
                #TP2 += 1
            elseif (df[i, "LABEL"] == 1 && df[i, "p(1)"] < thd)
                FN += (1 * 5.5496)
                #FN2 += 1
            elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] >= thd)
                FP += (1 * 0.5495)
                #FP2 += 1
            elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] < thd)
                TN += (1 * 0.5495)
                #TN2 += 1
            end
        end
        return (TP / (TP + FN)), (FN / (TP + FN)), (FP / (FP + TP)), (FP / (FP + TN)), (TN / (TN + FP)), (TP / (TP + FP)), (TP / (TP + FN))
    end
    #
    function get1ratePest(df, thd)
        TP = 0  # 
        FN = 0  # 
        TN = 0  # 
        FP = 0  # 
        #TP2 = 0
        #FN2 = 0
        #TN2 = 0
        #FP2 = 0
        for i in 1:size(df , 1)
            if (df[i, "LABEL"] == 1 && df[i, "p(1)"] >= thd)
                TP += (1 * 1.4602)
                #TP2 += 1
            elseif (df[i, "LABEL"] == 1 && df[i, "p(1)"] < thd)
                FN += (1 * 1.4602)
                #FN2 += 1
            elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] >= thd)
                FP += (1 * 0.7604)
                #FP2 += 1
            elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] < thd)
                TN += (1 * 0.7604)
                #TN2 += 1
            end
        end
        return (TP / (TP + FN)), (FN / (TP + FN)), (FP / (FP + TP)), (FP / (FP + TN)), (TN / (TN + FP)), (TP / (TP + FP)), (TP / (TP + FN))
    end
    #
    ## call function and insert arrays as columns ##
    TrainWithDeltaRi_TPR = []
    TrainWithDeltaRi_FNR = []
    TrainWithDeltaRi_FDR = []
    TrainWithDeltaRi_FPR = []
    TrainWithDeltaRi_TNR = []
    TrainWithDeltaRi_Precision = []
    TrainWithDeltaRi_Recall = []
    prob = 0
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    Precision = 0
    Recall = 0
    FPR_ = 0
    auroc = 0
    Recall_ = 0
    auprc = 0
    for temp in Array(inputDB_TrainWithDeltaRi[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR, Precision, Recall = get1rateTrain(inputDB_TrainWithDeltaRi, prob)
            push!(TrainWithDeltaRi_TPR, TPR)
            push!(TrainWithDeltaRi_FNR, FNR)
            push!(TrainWithDeltaRi_FDR, FDR)
            push!(TrainWithDeltaRi_FPR, FPR)
            push!(TrainWithDeltaRi_TNR, TNR)
            push!(TrainWithDeltaRi_Precision, Precision)
            push!(TrainWithDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        else
            push!(TrainWithDeltaRi_TPR, TPR)
            push!(TrainWithDeltaRi_FNR, FNR)
            push!(TrainWithDeltaRi_FDR, FDR)
            push!(TrainWithDeltaRi_FPR, FPR)
            push!(TrainWithDeltaRi_TNR, TNR)
            push!(TrainWithDeltaRi_Precision, Precision)
            push!(TrainWithDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        end
    end
    inputDB_TrainWithDeltaRi[!, "TPR"] = TrainWithDeltaRi_TPR
    inputDB_TrainWithDeltaRi[!, "FNR"] = TrainWithDeltaRi_FNR
    inputDB_TrainWithDeltaRi[!, "FDR"] = TrainWithDeltaRi_FDR
    inputDB_TrainWithDeltaRi[!, "FPR"] = TrainWithDeltaRi_FPR
    inputDB_TrainWithDeltaRi[!, "TNR"] = TrainWithDeltaRi_TNR
    inputDB_TrainWithDeltaRi[!, "Precision"] = TrainWithDeltaRi_Precision
    inputDB_TrainWithDeltaRi[!, "Recall"] = TrainWithDeltaRi_Recall
    print(auroc, " ", auprc)  #0.9485, 0.9322

## save ##, ouputing df 1686319 x 26+5 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newTrainALL_KNN.csv"
CSV.write(savePath, inputDB_TrainWithDeltaRi)


# ==================================================================================================
## prepare to plot P(TP)threshold-to-TPR curve ## testing set
    ## call function and insert arrays as columns ##
    TestWithDeltaRi_TPR = []
    TestWithDeltaRi_FNR = []
    TestWithDeltaRi_FDR = []
    TestWithDeltaRi_FPR = []
    TestWithDeltaRi_TNR = []
    TestWithDeltaRi_Precision = []
    TestWithDeltaRi_Recall = []
    prob = 0
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    Precision = 0
    Recall = 0
    FPR_ = 0
    auroc = 0
    Recall_ = 0
    auprc = 0
    for temp in Array(inputDB_TestWithDeltaRi[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR, Precision, Recall = get1rateTest(inputDB_TestWithDeltaRi, prob)
            push!(TestWithDeltaRi_TPR, TPR)
            push!(TestWithDeltaRi_FNR, FNR)
            push!(TestWithDeltaRi_FDR, FDR)
            push!(TestWithDeltaRi_FPR, FPR)
            push!(TestWithDeltaRi_TNR, TNR)
            push!(TestWithDeltaRi_Precision, Precision)
            push!(TestWithDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        else
            push!(TestWithDeltaRi_TPR, TPR)
            push!(TestWithDeltaRi_FNR, FNR)
            push!(TestWithDeltaRi_FDR, FDR)
            push!(TestWithDeltaRi_FPR, FPR)
            push!(TestWithDeltaRi_TNR, TNR)
            push!(TestWithDeltaRi_Precision, Precision)
            push!(TestWithDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        end
    end
    inputDB_TestWithDeltaRi[!, "TPR"] = TestWithDeltaRi_TPR
    inputDB_TestWithDeltaRi[!, "FNR"] = TestWithDeltaRi_FNR
    inputDB_TestWithDeltaRi[!, "FDR"] = TestWithDeltaRi_FDR
    inputDB_TestWithDeltaRi[!, "FPR"] = TestWithDeltaRi_FPR
    inputDB_TestWithDeltaRi[!, "TNR"] = TestWithDeltaRi_TNR
    inputDB_TestWithDeltaRi[!, "Precision"] = TestWithDeltaRi_Precision
    inputDB_TestWithDeltaRi[!, "Recall"] = TestWithDeltaRi_Recall
    print(auroc, " ", auprc)  #0.9461, 0.9289

## save ##, ouputing df 421381 x 26+5 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newTestALL_KNN.csv"
CSV.write(savePath, inputDB_TestWithDeltaRi)


# ==================================================================================================
## prepare to plot P(TP)threshold-to-TPR curve ## validation set (No Tea Spike Blank)
    ## call function and insert arrays as columns ##
    PestWithDeltaRi_TPR = []
    PestWithDeltaRi_FNR = []
    PestWithDeltaRi_FDR = []
    PestWithDeltaRi_FPR = []
    PestWithDeltaRi_TNR = []
    PestWithDeltaRi_Precision = []
    PestWithDeltaRi_Recall = []
    prob = 0
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    Precision = 0
    Recall = 0
    FPR_ = 0
    auroc = 0
    Recall_ = 0
    auprc = 0
    for temp in Array(inputDB_PestWithDeltaRi[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR, Precision, Recall = get1ratePest(inputDB_PestWithDeltaRi, prob)
            push!(PestWithDeltaRi_TPR, TPR)
            push!(PestWithDeltaRi_FNR, FNR)
            push!(PestWithDeltaRi_FDR, FDR)
            push!(PestWithDeltaRi_FPR, FPR)
            push!(PestWithDeltaRi_TNR, TNR)
            push!(PestWithDeltaRi_Precision, Precision)
            push!(PestWithDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        else
            push!(PestWithDeltaRi_TPR, TPR)
            push!(PestWithDeltaRi_FNR, FNR)
            push!(PestWithDeltaRi_FDR, FDR)
            push!(PestWithDeltaRi_FPR, FPR)
            push!(PestWithDeltaRi_TNR, TNR)
            push!(PestWithDeltaRi_Precision, Precision)
            push!(PestWithDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        end
    end
    inputDB_PestWithDeltaRi[!, "TPR"] = PestWithDeltaRi_TPR
    inputDB_PestWithDeltaRi[!, "FNR"] = PestWithDeltaRi_FNR
    inputDB_PestWithDeltaRi[!, "FDR"] = PestWithDeltaRi_FDR
    inputDB_PestWithDeltaRi[!, "FPR"] = PestWithDeltaRi_FPR
    inputDB_PestWithDeltaRi[!, "TNR"] = PestWithDeltaRi_TNR
    inputDB_PestWithDeltaRi[!, "Precision"] = PestWithDeltaRi_Precision
    inputDB_PestWithDeltaRi[!, "Recall"] = PestWithDeltaRi_Recall
    print(auroc, " ", auprc)  # 0.7328, 0.6301

## save ##, ouputing df 10908 x 23+5 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newPestNoTea_KNN.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)


# ==================================================================================================
## input ## 1686319 x 25 df; 421381 x 25 df; 10908 x 22 df; 8187 x 22 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage, withDeltaRipredictTPTN, p0, p1
inputDB_TrainWithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_TrainDF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_TestWithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_ValDF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_PestWithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_PestDF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_Pest2WithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_Pest2DF_withOutDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)

## prepare to plot confusion matrix for training set ##
inputDB_TrainWithOutDeltaRi[!, "CM"] .= String("")
    inputDB_TrainWithOutDeltaRi_TP = 0
    inputDB_TrainWithOutDeltaRi_FP = 0
    inputDB_TrainWithOutDeltaRi_TN = 0
    inputDB_TrainWithOutDeltaRi_FN = 0
    for i in 1:size(inputDB_TrainWithOutDeltaRi , 1)
        if (inputDB_TrainWithOutDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
            inputDB_TrainWithOutDeltaRi[i, "CM"] = "TP"
            inputDB_TrainWithOutDeltaRi_TP += 1
        elseif (inputDB_TrainWithOutDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
            inputDB_TrainWithOutDeltaRi[i, "CM"] = "FP"
            inputDB_TrainWithOutDeltaRi_FP += 1
        elseif (inputDB_TrainWithOutDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
            inputDB_TrainWithOutDeltaRi[i, "CM"] = "TN"
            inputDB_TrainWithOutDeltaRi_TN += 1
        elseif (inputDB_TrainWithOutDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
            inputDB_TrainWithOutDeltaRi[i, "CM"] = "FN"
            inputDB_TrainWithOutDeltaRi_FN += 1
        end
    end
    #
    CM_TrainWithOut = zeros(2, 2)
    CM_TrainWithOut[2, 1] = inputDB_TrainWithOutDeltaRi_TP  #150,652
    CM_TrainWithOut[2, 2] = inputDB_TrainWithOutDeltaRi_FP  #350,380
    CM_TrainWithOut[1, 2] = inputDB_TrainWithOutDeltaRi_TN  #1,184,629
    CM_TrainWithOut[1, 1] = inputDB_TrainWithOutDeltaRi_FN  #658

## save ##, ouputing df 1686319 x 25+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TrainALLWithOutDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_TrainWithOutDeltaRi)


# ==================================================================================================
## prepare to plot confusion matrix for testing set ##
inputDB_TestWithOutDeltaRi[!, "CM"] .= String("")
    inputDB_TestWithOutDeltaRi_TP = 0
    inputDB_TestWithOutDeltaRi_FP = 0
    inputDB_TestWithOutDeltaRi_TN = 0
    inputDB_TestWithOutDeltaRi_FN = 0
    for i in 1:size(inputDB_TestWithOutDeltaRi , 1)
        if (inputDB_TestWithOutDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
            inputDB_TestWithOutDeltaRi[i, "CM"] = "TP"
            inputDB_TestWithOutDeltaRi_TP += 1
        elseif (inputDB_TestWithOutDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
            inputDB_TestWithOutDeltaRi[i, "CM"] = "FP"
            inputDB_TestWithOutDeltaRi_FP += 1
        elseif (inputDB_TestWithOutDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
            inputDB_TestWithOutDeltaRi[i, "CM"] = "TN"
            inputDB_TestWithOutDeltaRi_TN += 1
        elseif (inputDB_TestWithOutDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
            inputDB_TestWithOutDeltaRi[i, "CM"] = "FN"
            inputDB_TestWithOutDeltaRi_FN += 1
        end
    end
    #
    CM_TestWithOut = zeros(2, 2)
    CM_TestWithOut[2, 1] = inputDB_TestWithOutDeltaRi_TP  #37,766
    CM_TestWithOut[2, 2] = inputDB_TestWithOutDeltaRi_FP  #88,117
    CM_TestWithOut[1, 2] = inputDB_TestWithOutDeltaRi_TN  #295,299
    CM_TestWithOut[1, 1] = inputDB_TestWithOutDeltaRi_FN  #199

## save ##, ouputing df 421381 x 25+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TestALLWithOutDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_TestWithOutDeltaRi)


# ==================================================================================================
## prepare to plot confusion matrix for validation set ##, No Tea spike blank
inputDB_PestWithOutDeltaRi[!, "CM"] .= String("")
    inputDB_PestWithOutDeltaRi_TP = 0
    inputDB_PestWithOutDeltaRi_FP = 0
    inputDB_PestWithOutDeltaRi_TN = 0
    inputDB_PestWithOutDeltaRi_FN = 0
    for i in 1:size(inputDB_PestWithOutDeltaRi , 1)
        if (inputDB_PestWithOutDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
            inputDB_PestWithOutDeltaRi[i, "CM"] = "TP"
            inputDB_PestWithOutDeltaRi_TP += 1
        elseif (inputDB_PestWithOutDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
            inputDB_PestWithOutDeltaRi[i, "CM"] = "FP"
            inputDB_PestWithOutDeltaRi_FP += 1
        elseif (inputDB_PestWithOutDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
            inputDB_PestWithOutDeltaRi[i, "CM"] = "TN"
            inputDB_PestWithOutDeltaRi_TN += 1
        elseif (inputDB_PestWithOutDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
            inputDB_PestWithOutDeltaRi[i, "CM"] = "FN"
            inputDB_PestWithOutDeltaRi_FN += 1
        end
    end
    #
    CM_PestWithOut = zeros(2, 2)
    CM_PestWithOut[2, 1] = inputDB_PestWithOutDeltaRi_TP  #2,457
    CM_PestWithOut[2, 2] = inputDB_PestWithOutDeltaRi_FP  #2,543
    CM_PestWithOut[1, 2] = inputDB_PestWithOutDeltaRi_TN  #4,630
    CM_PestWithOut[1, 1] = inputDB_PestWithOutDeltaRi_FN  #1,278

## save ##, ouputing df 10908 x 22+1 df
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_PestNoTeaWithOutDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_PestWithOutDeltaRi)


# ==================================================================================================
## prepare to plot confusion matrix for real sample set ##, With Tea
inputDB_Pest2WithOutDeltaRi[!, "CM"] .= String("")
    inputDB_Pest2WithOutDeltaRi_TP = 0
    inputDB_Pest2WithOutDeltaRi_FP = 0
    inputDB_Pest2WithOutDeltaRi_TN = 0
    inputDB_Pest2WithOutDeltaRi_FN = 0
    for i in 1:size(inputDB_Pest2WithOutDeltaRi , 1)
        if (inputDB_Pest2WithOutDeltaRi[i, "LABEL"] == 1 && inputDB_Pest2WithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
            inputDB_Pest2WithOutDeltaRi[i, "CM"] = "TP"
            inputDB_Pest2WithOutDeltaRi_TP += 1
        elseif (inputDB_Pest2WithOutDeltaRi[i, "LABEL"] == 0 && inputDB_Pest2WithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 1)
            inputDB_Pest2WithOutDeltaRi[i, "CM"] = "FP"
            inputDB_Pest2WithOutDeltaRi_FP += 1
        elseif (inputDB_Pest2WithOutDeltaRi[i, "LABEL"] == 0 && inputDB_Pest2WithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
            inputDB_Pest2WithOutDeltaRi[i, "CM"] = "TN"
            inputDB_Pest2WithOutDeltaRi_TN += 1
        elseif (inputDB_Pest2WithOutDeltaRi[i, "LABEL"] == 1 && inputDB_Pest2WithOutDeltaRi[i, "withoutDeltaRipredictTPTN"] == 0)
            inputDB_Pest2WithOutDeltaRi[i, "CM"] = "FN"
            inputDB_Pest2WithOutDeltaRi_FN += 1
        end
    end
    #
    CM_Pest2WithOut = zeros(2, 2)
    CM_Pest2WithOut[2, 1] = inputDB_Pest2WithOutDeltaRi_TP  #4,445
    CM_Pest2WithOut[2, 2] = inputDB_Pest2WithOutDeltaRi_FP  #0
    CM_Pest2WithOut[1, 2] = inputDB_Pest2WithOutDeltaRi_TN  #0
    CM_Pest2WithOut[1, 1] = inputDB_Pest2WithOutDeltaRi_FN  #3,742

## save ##, ouputing df 8187 x 22+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_Pest2WithTeaWithOutDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_Pest2WithOutDeltaRi)

## prepare to plot P(TP)threshold-to-TPR curve ## training set
    ##  1686319 x 26 df
    inputDB_TrainWithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TrainALLWithOutDeltaRI_KNN.csv", DataFrame)
        sort!(inputDB_TrainWithOutDeltaRi, [:"p(1)"], rev = true)
        for i in 1:size(inputDB_TrainWithOutDeltaRi, 1)
            inputDB_TrainWithOutDeltaRi[i, "p(1)"] = round(float(inputDB_TrainWithOutDeltaRi[i, "p(1)"]), digits = 2)
        end
        #
    # 421381 x 26 df
    inputDB_TestWithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TestALLWithOutDeltaRI_KNN.csv", DataFrame)
        sort!(inputDB_TestWithOutDeltaRi, [:"p(1)"], rev = true)
        for i in 1:size(inputDB_TestWithOutDeltaRi, 1)
            inputDB_TestWithOutDeltaRi[i, "p(1)"] = round(float(inputDB_TestWithOutDeltaRi[i, "p(1)"]), digits = 2)
        end
        #
    # 10908 x 23 df
    inputDB_PestWithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_PestNoTeaWithOutDeltaRI_KNN.csv", DataFrame)
        sort!(inputDB_PestWithOutDeltaRi, [:"p(1)"], rev = true)
        for i in 1:size(inputDB_PestWithOutDeltaRi, 1)
            inputDB_PestWithOutDeltaRi[i, "p(1)"] = round(float(inputDB_PestWithOutDeltaRi[i, "p(1)"]), digits = 2)
        end
        #
    # 8187 x 23 df
    inputDB_Pest2WithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_Pest2WithTeaWithOutDeltaRI_KNN.csv", DataFrame)
        sort!(inputDB_Pest2WithOutDeltaRi, [:"p(1)"], rev = true)
        for i in 1:size(inputDB_Pest2WithOutDeltaRi, 1)
            inputDB_Pest2WithOutDeltaRi[i, "p(1)"] = round(float(inputDB_Pest2WithOutDeltaRi[i, "p(1)"]), digits = 2)
        end
    #
    #
    ## call function and insert arrays as columns ##
    TrainWithOutDeltaRi_TPR = []
    TrainWithOutDeltaRi_FNR = []
    TrainWithOutDeltaRi_FDR = []
    TrainWithOutDeltaRi_FPR = []
    TrainWithOutDeltaRi_TNR = []
    TrainWithOutDeltaRi_Precision = []
    TrainWithOutDeltaRi_Recall = []
    prob = 0
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    Precision = 0
    Recall = 0
    FPR_ = 0
    auroc = 0
    Recall_ = 0
    auprc = 0
    for temp in Array(inputDB_TrainWithOutDeltaRi[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR, Precision, Recall = get1rateTrain(inputDB_TrainWithOutDeltaRi, prob)
            push!(TrainWithOutDeltaRi_TPR, TPR)
            push!(TrainWithOutDeltaRi_FNR, FNR)
            push!(TrainWithOutDeltaRi_FDR, FDR)
            push!(TrainWithOutDeltaRi_FPR, FPR)
            push!(TrainWithOutDeltaRi_TNR, TNR)
            push!(TrainWithOutDeltaRi_Precision, Precision)
            push!(TrainWithOutDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        else
            push!(TrainWithOutDeltaRi_TPR, TPR)
            push!(TrainWithOutDeltaRi_FNR, FNR)
            push!(TrainWithOutDeltaRi_FDR, FDR)
            push!(TrainWithOutDeltaRi_FPR, FPR)
            push!(TrainWithOutDeltaRi_TNR, TNR)
            push!(TrainWithOutDeltaRi_Precision, Precision)
            push!(TrainWithOutDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        end
    end
    inputDB_TrainWithOutDeltaRi[!, "TPR"] = TrainWithOutDeltaRi_TPR
    inputDB_TrainWithOutDeltaRi[!, "FNR"] = TrainWithOutDeltaRi_FNR
    inputDB_TrainWithOutDeltaRi[!, "FDR"] = TrainWithOutDeltaRi_FDR
    inputDB_TrainWithOutDeltaRi[!, "FPR"] = TrainWithOutDeltaRi_FPR
    inputDB_TrainWithOutDeltaRi[!, "TNR"] = TrainWithOutDeltaRi_TNR
    inputDB_TrainWithOutDeltaRi[!, "Precision"] = TrainWithOutDeltaRi_Precision
    inputDB_TrainWithOutDeltaRi[!, "Recall"] = TrainWithOutDeltaRi_Recall
    print(auroc, " ", auprc)  #0.9414, 0.9171

## save ##, ouputing df 1686319 x 26+5 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newTrainALLwithOut_KNN.csv"
CSV.write(savePath, inputDB_TrainWithOutDeltaRi)


# ==================================================================================================
## prepare to plot P(TP)threshold-to-TPR curve ## testing set
    ## call function and insert arrays as columns ##
    TestWithOutDeltaRi_TPR = []
    TestWithOutDeltaRi_FNR = []
    TestWithOutDeltaRi_FDR = []
    TestWithOutDeltaRi_FPR = []
    TestWithOutDeltaRi_TNR = []
    TestWithOutDeltaRi_Precision = []
    TestWithOutDeltaRi_Recall = []
    prob = -1
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    Precision = 0
    Recall = 0
    FPR_ = 0
    auroc = 0
    Recall_ = 0
    auprc = 0
    for temp in Array(inputDB_TestWithOutDeltaRi[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR, Precision, Recall = get1rateTest(inputDB_TestWithOutDeltaRi, prob)
            push!(TestWithOutDeltaRi_TPR, TPR)
            push!(TestWithOutDeltaRi_FNR, FNR)
            push!(TestWithOutDeltaRi_FDR, FDR)
            push!(TestWithOutDeltaRi_FPR, FPR)
            push!(TestWithOutDeltaRi_TNR, TNR)
            push!(TestWithOutDeltaRi_Precision, Precision)
            push!(TestWithOutDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        else
            push!(TestWithOutDeltaRi_TPR, TPR)
            push!(TestWithOutDeltaRi_FNR, FNR)
            push!(TestWithOutDeltaRi_FDR, FDR)
            push!(TestWithOutDeltaRi_FPR, FPR)
            push!(TestWithOutDeltaRi_TNR, TNR)
            push!(TestWithOutDeltaRi_Precision, Precision)
            push!(TestWithOutDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        end
    end
    inputDB_TestWithOutDeltaRi[!, "TPR"] = TestWithOutDeltaRi_TPR
    inputDB_TestWithOutDeltaRi[!, "FNR"] = TestWithOutDeltaRi_FNR
    inputDB_TestWithOutDeltaRi[!, "FDR"] = TestWithOutDeltaRi_FDR
    inputDB_TestWithOutDeltaRi[!, "FPR"] = TestWithOutDeltaRi_FPR
    inputDB_TestWithOutDeltaRi[!, "TNR"] = TestWithOutDeltaRi_TNR
    inputDB_TestWithOutDeltaRi[!, "Precision"] = TestWithOutDeltaRi_Precision
    inputDB_TestWithOutDeltaRi[!, "Recall"] = TestWithOutDeltaRi_Recall
    print(auroc, " ", auprc)  #0.9381, 0.9118

## save ##, ouputing df 421381 x 26+5 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newTestALLwithOut_KNN.csv"
CSV.write(savePath, inputDB_TestWithOutDeltaRi)


# ==================================================================================================
## prepare to plot P(TP)threshold-to-TPR curve ## validation set (No Tea Spike Blank)
    ## call function and insert arrays as columns ##
    PestWithOutDeltaRi_TPR = []
    PestWithOutDeltaRi_FNR = []
    PestWithOutDeltaRi_FDR = []
    PestWithOutDeltaRi_FPR = []
    PestWithOutDeltaRi_TNR = []
    PestWithOutDeltaRi_Precision = []
    PestWithOutDeltaRi_Recall = []
    prob = -1
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    Precision = 0
    Recall = 0
    FPR_ = 0
    auroc = 0
    Recall_ = 0
    auprc = 0
    for temp in Array(inputDB_PestWithOutDeltaRi[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR, Precision, Recall = get1ratePest(inputDB_PestWithOutDeltaRi, prob)
            push!(PestWithOutDeltaRi_TPR, TPR)
            push!(PestWithOutDeltaRi_FNR, FNR)
            push!(PestWithOutDeltaRi_FDR, FDR)
            push!(PestWithOutDeltaRi_FPR, FPR)
            push!(PestWithOutDeltaRi_TNR, TNR)
            push!(PestWithOutDeltaRi_Precision, Precision)
            push!(PestWithOutDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        else
            push!(PestWithOutDeltaRi_TPR, TPR)
            push!(PestWithOutDeltaRi_FNR, FNR)
            push!(PestWithOutDeltaRi_FDR, FDR)
            push!(PestWithOutDeltaRi_FPR, FPR)
            push!(PestWithOutDeltaRi_TNR, TNR)
            push!(PestWithOutDeltaRi_Precision, Precision)
            push!(PestWithOutDeltaRi_Recall, Recall)
            auroc += (TPR*(FPR-FPR_))
            FPR_ = FPR
            auprc += (Precision*(Recall-Recall_))
            Recall_ = Recall
        end
    end
    inputDB_PestWithOutDeltaRi[!, "TPR"] = PestWithOutDeltaRi_TPR
    inputDB_PestWithOutDeltaRi[!, "FNR"] = PestWithOutDeltaRi_FNR
    inputDB_PestWithOutDeltaRi[!, "FDR"] = PestWithOutDeltaRi_FDR
    inputDB_PestWithOutDeltaRi[!, "FPR"] = PestWithOutDeltaRi_FPR
    inputDB_PestWithOutDeltaRi[!, "TNR"] = PestWithOutDeltaRi_TNR
    inputDB_PestWithOutDeltaRi[!, "Precision"] = PestWithOutDeltaRi_Precision
    inputDB_PestWithOutDeltaRi[!, "Recall"] = PestWithOutDeltaRi_Recall
    print(auroc, " ", auprc)  #0.7577, 0.6557

## save ##, ouputing df 10908 x 23+5 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newPestNoTeaWithOut_KNN.csv"
CSV.write(savePath, inputDB_PestWithOutDeltaRi)


# ==================================================================================================
## plot P(1)threshold-to-TPR & P(1)threshold-to-TNR ## for training set
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
TrainOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)
plot!(inputDB_TrainWithDeltaRi[:, end-8], [inputDB_TrainWithDeltaRi[:, end-6] inputDB_TrainWithDeltaRi[:, end-5]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=14, 
        label = ["True Positive Rate" "False Negative Rate"], 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        legend = :left, 
        legendfont = font(12), 
        linewidth = 2, 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_TrainWithDeltaRi[:, end-8], inputDB_TrainWithDeltaRi[:, end-4], 
        subplot = 2, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=14, 
        label = "False Discovery Rate", 
        yguidefontsize=14, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        ylims = [0, 0.23323328569055], 
        legend = :best, 
        legendfont = font(12), 
        linewidth = 2, 
        size = (1200,600), 
        dpi = 300)
        new_yticks = ([0.05], ["\$\\bar"], ["purple"])
        new_yticks2 = ([0.10], ["\$\\bar"], ["red"])
        hline!(new_yticks[1], label = "5% FDR-Controlled Cutoff at P(1) = 0.89", legendfont = font(12), lc = "purple", subplot = 2)
        hline!(new_yticks2[1], label = "10% FDR-Controlled Cutoff at P(1) = 0.78", legendfont = font(12), lc = "red", subplot = 2)
savefig(TrainOutplotP1toRate, "F:\\UvA\\results\\TPTNPrediction_P1threshold2TPRFNRFDR_newTrainALLylimsWith_KNN.png")


# ==================================================================================================
## input ## 1686319 x 25 df; 421381 x 25 df; 10908 x 22 df; 8187 x 22 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage, withDeltaRipredictTPTN, p0, p1
inputDB_TrainWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_TestWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_ValDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_PestDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_Pest2WithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_Pest2DF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)


# ==================================================================================================
inputDB_TrainWithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newTrainALLwithOut_KNN.csv", DataFrame)
inputDB_TestWithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newTestALLwithOut_KNN.csv", DataFrame)
inputDB_PestWithOutDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newPestNoTeaWithOut_KNN.csv", DataFrame)

## plot P(1)threshold-to-TPR & P(1)threshold-to-TNR ## for training set
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
TrainOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)
plot!(inputDB_PestWithDeltaRi[:, end-8], [inputDB_PestWithDeltaRi[:, end-1] inputDB_PestWithDeltaRi[:, end]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=14, 
        label = ["Precision" "Recall"], 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        legend = :left, 
        legendfont = font(12), 
        linewidth = 2, 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_PestWithOutDeltaRi[:, end-8], [inputDB_PestWithOutDeltaRi[:, end-1] inputDB_PestWithOutDeltaRi[:, end]], 
        subplot = 2, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=14, 
        label = ["Precision" "Recall"], 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        legend = :left, 
        legendfont = font(12), 
        linewidth = 2, 
        size = (1200,600), 
        dpi = 300)
savefig(TrainOutplotP1toRate, "F:\\UvA\\results\\PRthresh_Pest.png")

## plot P(1)threshold-to-TPR & P(1)threshold-to-TNR ## for training set
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
tempDfTest = inputDB_PestWithOutDeltaRi[:, end-3]
TrainOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)

inputDB_PestWithOutDeltaRi[:, end-8:end]
plot!(tempDfTest -> tempDfTest, c=:red, subplot = 1, 
        label = "Identity Line", 
        #margin = -2Plots.px, 
        size = (1200,600), 
        alpha = 0.75, 
        dpi = 300)
plot!(inputDB_PestWithOutDeltaRi[:, end-3], inputDB_PestWithOutDeltaRi[:, end-6], 
        subplot = 1, framestyle = :box, 
        xlabel = "False Postive Rate", 
        ylabel = "True Positive Rate", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "Without deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "orange", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_PestWithDeltaRi[:, end-3], inputDB_PestWithDeltaRi[:, end-6], 
        subplot = 1, framestyle = :box, 
        xlabel = "False Positive Rate", 
        ylabel = "True Positive Rate", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "With deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "purple", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)

inputDB_PestWithOutDeltaRi[:, end-8:end]

plot!(inputDB_PestWithOutDeltaRi[:, end], inputDB_PestWithOutDeltaRi[:, end-1], 
        subplot = 2, framestyle = :box, 
        xlabel = "Recall", 
        ylabel = "Precision", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "Without deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "orange", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_PestWithDeltaRi[:, end], inputDB_PestWithDeltaRi[:, end-1], 
        subplot = 2, framestyle = :box, 
        xlabel = "Recall", 
        ylabel = "Precision", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "With deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "purple", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)
savefig(TrainOutplotP1toRate, "F:\\UvA\\results\\PRcurve_Pest.png")


# ==================================================================================================
## plot P(1)threshold-to-TPR & P(1)threshold-to-TNR ## for training set
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
tempDfTest = inputDB_PestWithOutDeltaRi[:, end-3]
TrainOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)

inputDB_PestWithOutDeltaRi[:, end-8:end]
plot!(tempDfTest -> tempDfTest, c=:red, subplot = 1, 
        label = "Identity Line", 
        #margin = -2Plots.px, 
        size = (1200,600), 
        alpha = 0.75, 
        dpi = 300)
plot!(tempDfTest -> tempDfTest, c=:red, subplot = 2, 
        label = "Identity Line", 
        #margin = -2Plots.px, 
        size = (1200,600), 
        alpha = 0.75, 
        dpi = 300)
plot!(inputDB_TrainWithOutDeltaRi[:, end-3], inputDB_TrainWithOutDeltaRi[:, end-6], 
        subplot = 1, framestyle = :box, 
        xlabel = "False Postive Rate", 
        ylabel = "True Positive Rate", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "Training without deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "orange", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_TrainWithDeltaRi[:, end-3], inputDB_TrainWithDeltaRi[:, end-6], 
        subplot = 1, framestyle = :box, 
        xlabel = "False Positive Rate", 
        ylabel = "True Positive Rate", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "Training with deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "purple", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_TestWithOutDeltaRi[:, end-3], inputDB_TestWithOutDeltaRi[:, end-6], 
        subplot = 1, framestyle = :box, 
        xlabel = "False Postive Rate", 
        ylabel = "True Positive Rate", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "Testing without deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "green", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_TestWithDeltaRi[:, end-3], inputDB_TestWithDeltaRi[:, end-6], 
        subplot = 1, framestyle = :box, 
        xlabel = "False Positive Rate", 
        ylabel = "True Positive Rate", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "Testing with deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "blue", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_PestWithOutDeltaRi[:, end-3], inputDB_PestWithOutDeltaRi[:, end-6], 
        subplot = 2, framestyle = :box, 
        xlabel = "False Postive Rate", 
        ylabel = "True Positive Rate", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "Testing without deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "green", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_PestWithDeltaRi[:, end-3], inputDB_PestWithDeltaRi[:, end-6], 
        subplot = 2, framestyle = :box, 
        xlabel = "False Positive Rate", 
        ylabel = "True Positive Rate", 
        xguidefontsize=14, 
        yguidefontsize=14, 
        label = "Testing with deltaRI", 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        linewidth = 2, 
        color = "blue", 
        alpha = 0.75, 
        legend = :bottomright, 
        legendfont = font(12), 
        size = (1200,600), 
        dpi = 300)
savefig(TrainOutplotP1toRate, "F:\\UvA\\results\\AUROCcurve_TrainTestPest.png")

# ==================================================================================================
## prepare to plot confusion matrix for training set ## 10% FDR controlled threshold- 0.78
inputDB_TrainWithDeltaRi[!, "CM"] .= String("")
    inputDB_TrainWithDeltaRi_TP = 0
    inputDB_TrainWithDeltaRi_FP = 0
    inputDB_TrainWithDeltaRi_TN = 0
    inputDB_TrainWithDeltaRi_FN = 0
    for i in 1:size(inputDB_TrainWithDeltaRi , 1)
        if (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "p(1)"] >= 0.78)
            inputDB_TrainWithDeltaRi[i, "CM"] = "TP"
            inputDB_TrainWithDeltaRi_TP += 1
        elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "p(1)"] >= 0.78)
            inputDB_TrainWithDeltaRi[i, "CM"] = "FP"
            inputDB_TrainWithDeltaRi_FP += 1
        elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 0 && inputDB_TrainWithDeltaRi[i, "p(1)"] < 0.78)
            inputDB_TrainWithDeltaRi[i, "CM"] = "TN"
            inputDB_TrainWithDeltaRi_TN += 1
        elseif (inputDB_TrainWithDeltaRi[i, "LABEL"] == 1 && inputDB_TrainWithDeltaRi[i, "p(1)"] < 0.78)
            inputDB_TrainWithDeltaRi[i, "CM"] = "FN"
            inputDB_TrainWithDeltaRi_FN += 1
        end
    end
    #
    CM_TrainWith = zeros(2, 2)
    CM_TrainWith[2, 1] = inputDB_TrainWithDeltaRi_TP
    CM_TrainWith[2, 2] = inputDB_TrainWithDeltaRi_FP
    CM_TrainWith[1, 2] = inputDB_TrainWithDeltaRi_TN
    CM_TrainWith[1, 1] = inputDB_TrainWithDeltaRi_FN

## save ##, ouputing df 1686319 x 25+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict10FDR_TrainALLWithDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_TrainWithDeltaRi)


# ==================================================================================================
## prepare to plot confusion matrix for testing set ## 10% FDR controlled threshold- 0.78
inputDB_TestWithDeltaRi[!, "CM"] .= String("")
    inputDB_TestWithDeltaRi_TP = 0
    inputDB_TestWithDeltaRi_FP = 0
    inputDB_TestWithDeltaRi_TN = 0
    inputDB_TestWithDeltaRi_FN = 0
    for i in 1:size(inputDB_TestWithDeltaRi , 1)
        if (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "p(1)"] >= 0.78)
            inputDB_TestWithDeltaRi[i, "CM"] = "TP"
            inputDB_TestWithDeltaRi_TP += 1
        elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "p(1)"] >= 0.78)
            inputDB_TestWithDeltaRi[i, "CM"] = "FP"
            inputDB_TestWithDeltaRi_FP += 1
        elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 0 && inputDB_TestWithDeltaRi[i, "p(1)"] < 0.78)
            inputDB_TestWithDeltaRi[i, "CM"] = "TN"
            inputDB_TestWithDeltaRi_TN += 1
        elseif (inputDB_TestWithDeltaRi[i, "LABEL"] == 1 && inputDB_TestWithDeltaRi[i, "p(1)"] < 0.78)
            inputDB_TestWithDeltaRi[i, "CM"] = "FN"
            inputDB_TestWithDeltaRi_FN += 1
        end
    end
    #
    CM_TestWith = zeros(2, 2)
    CM_TestWith[2, 1] = inputDB_TestWithDeltaRi_TP
    CM_TestWith[2, 2] = inputDB_TestWithDeltaRi_FP
    CM_TestWith[1, 2] = inputDB_TestWithDeltaRi_TN
    CM_TestWith[1, 1] = inputDB_TestWithDeltaRi_FN

## save ##, ouputing df 421381 x 25+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict10FDR_TestALLWithDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_TestWithDeltaRi)


# ==================================================================================================
## prepare to plot confusion matrix for validation set (No Tea Spike Blank) ## 10% FDR controlled threshold- 0.78
inputDB_PestWithDeltaRi[!, "CM"] .= String("")
    inputDB_PestWithDeltaRi_TP = 0
    inputDB_PestWithDeltaRi_FP = 0
    inputDB_PestWithDeltaRi_TN = 0
    inputDB_PestWithDeltaRi_FN = 0
    for i in 1:size(inputDB_PestWithDeltaRi , 1)
        if (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "p(1)"] >= 0.78)
            inputDB_PestWithDeltaRi[i, "CM"] = "TP"
            inputDB_PestWithDeltaRi_TP += 1
        elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "p(1)"] >= 0.78)
            inputDB_PestWithDeltaRi[i, "CM"] = "FP"
            inputDB_PestWithDeltaRi_FP += 1
        elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 0 && inputDB_PestWithDeltaRi[i, "p(1)"] < 0.78)
            inputDB_PestWithDeltaRi[i, "CM"] = "TN"
            inputDB_PestWithDeltaRi_TN += 1
        elseif (inputDB_PestWithDeltaRi[i, "LABEL"] == 1 && inputDB_PestWithDeltaRi[i, "p(1)"] < 0.78)
            inputDB_PestWithDeltaRi[i, "CM"] = "FN"
            inputDB_PestWithDeltaRi_FN += 1
        end
    end
    #
    CM_PestWith = zeros(2, 2)
    CM_PestWith[2, 1] = inputDB_PestWithDeltaRi_TP
    CM_PestWith[2, 2] = inputDB_PestWithDeltaRi_FP
    CM_PestWith[1, 2] = inputDB_PestWithDeltaRi_TN
    CM_PestWith[1, 1] = inputDB_PestWithDeltaRi_FN

## save ##, ouputing df 10908 x 22+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict10FDR_PestNoTeaWithDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)


# ==================================================================================================
## prepare to plot confusion matrix for real sample set (With Tea) ## 10% FDR controlled threshold- 0.78
inputDB_Pest2WithDeltaRi[!, "CM"] .= String("")
    inputDB_Pest2WithDeltaRi_TP = 0
    inputDB_Pest2WithDeltaRi_FP = 0
    inputDB_Pest2WithDeltaRi_TN = 0
    inputDB_Pest2WithDeltaRi_FN = 0
    for i in 1:size(inputDB_Pest2WithDeltaRi , 1)
        if (inputDB_Pest2WithDeltaRi[i, "LABEL"] == 1 && inputDB_Pest2WithDeltaRi[i, "p(1)"] >= 0.78)
            inputDB_Pest2WithDeltaRi[i, "CM"] = "TP"
            inputDB_Pest2WithDeltaRi_TP += 1
        elseif (inputDB_Pest2WithDeltaRi[i, "LABEL"] == 0 && inputDB_Pest2WithDeltaRi[i, "p(1)"] >= 0.78)
            inputDB_Pest2WithDeltaRi[i, "CM"] = "FP"
            inputDB_Pest2WithDeltaRi_FP += 1
        elseif (inputDB_Pest2WithDeltaRi[i, "LABEL"] == 0 && inputDB_Pest2WithDeltaRi[i, "p(1)"] < 0.78)
            inputDB_Pest2WithDeltaRi[i, "CM"] = "TN"
            inputDB_Pest2WithDeltaRi_TN += 1
        elseif (inputDB_Pest2WithDeltaRi[i, "LABEL"] == 1 && inputDB_Pest2WithDeltaRi[i, "p(1)"] < 0.78)
            inputDB_Pest2WithDeltaRi[i, "CM"] = "FN"
            inputDB_Pest2WithDeltaRi_FN += 1
        end
    end
    #
    CM_Pest2With = zeros(2, 2)
    CM_Pest2With[2, 1] = inputDB_Pest2WithDeltaRi_TP
    CM_Pest2With[2, 2] = inputDB_Pest2WithDeltaRi_FP
    CM_Pest2With[1, 2] = inputDB_Pest2WithDeltaRi_TN
    CM_Pest2With[1, 1] = inputDB_Pest2WithDeltaRi_FN

## save ##, ouputing df 10908 x 22+1 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict10FDR_Pest2WithTeaWithDeltaRI_KNN.csv"
CSV.write(savePath, inputDB_Pest2WithDeltaRi)
