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
    CM_TrainWith[2, 1] = inputDB_TrainWithDeltaRi_TP
    CM_TrainWith[2, 2] = inputDB_TrainWithDeltaRi_FP
    CM_TrainWith[1, 2] = inputDB_TrainWithDeltaRi_TN
    CM_TrainWith[1, 1] = inputDB_TrainWithDeltaRi_FN

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
    CM_TestWith[2, 1] = inputDB_TestWithDeltaRi_TP
    CM_TestWith[2, 2] = inputDB_TestWithDeltaRi_FP
    CM_TestWith[1, 2] = inputDB_TestWithDeltaRi_TN
    CM_TestWith[1, 1] = inputDB_TestWithDeltaRi_FN

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
    CM_PestWith[2, 1] = inputDB_PestWithDeltaRi_TP
    CM_PestWith[2, 2] = inputDB_PestWithDeltaRi_FP
    CM_PestWith[1, 2] = inputDB_PestWithDeltaRi_TN
    CM_PestWith[1, 1] = inputDB_PestWithDeltaRi_FN

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
    CM_Pest2With[2, 1] = inputDB_Pest2WithDeltaRi_TP
    CM_Pest2With[2, 2] = inputDB_Pest2WithDeltaRi_FP
    CM_Pest2With[1, 2] = inputDB_Pest2WithDeltaRi_TN
    CM_Pest2With[1, 1] = inputDB_Pest2WithDeltaRi_FN

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
    function get1rate(df, thd)
        TP = 0  # 
        FN = 0  # 
        TN = 0  # 
        FP = 0  # 
        for i in 1:size(df , 1)
            if (df[i, "LABEL"] == 1 && df[i, "p(1)"] >= thd)
                TP += (1 * 5.5724)
            elseif (df[i, "LABEL"] == 1 && df[i, "p(1)"] < thd)
                FN += (1 * 5.5724)
            elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] >= thd)
                FP += (1 * 0.5493)
            elseif (df[i, "LABEL"] == 0 && df[i, "p(1)"] < thd)
                TN += (1 * 0.5493)
            end
        end
        return (TP / (TP + FN)), (FN / (TP + FN)), (FP / (FP + TP)), (FP / (FP + TN)), (TN / (TN + FP))
    end
    #
    ## call function and insert arrays as columns ##
    TrainWithDeltaRi_TPR = []
    TrainWithDeltaRi_FNR = []
    TrainWithDeltaRi_FDR = []
    TrainWithDeltaRi_FPR = []
    TrainWithDeltaRi_TNR = []
    prob = -1
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    for temp in Array(inputDB_TrainWithDeltaRi[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR = get1rate(inputDB_TrainWithDeltaRi, prob)
            push!(TrainWithDeltaRi_TPR, TPR)
            push!(TrainWithDeltaRi_FNR, FNR)
            push!(TrainWithDeltaRi_FDR, FDR)
            push!(TrainWithDeltaRi_FPR, FPR)
            push!(TrainWithDeltaRi_TNR, TNR)
        else
            push!(TrainWithDeltaRi_TPR, TPR)
            push!(TrainWithDeltaRi_FNR, FNR)
            push!(TrainWithDeltaRi_FDR, FDR)
            push!(TrainWithDeltaRi_FPR, FPR)
            push!(TrainWithDeltaRi_TNR, TNR)
        end
    end
    inputDB_TrainWithDeltaRi[!, "TPR"] = TrainWithDeltaRi_TPR
    inputDB_TrainWithDeltaRi[!, "FNR"] = TrainWithDeltaRi_FNR
    inputDB_TrainWithDeltaRi[!, "FDR"] = TrainWithDeltaRi_FDR
    inputDB_TrainWithDeltaRi[!, "FPR"] = TrainWithDeltaRi_FPR
    inputDB_TrainWithDeltaRi[!, "TNR"] = TrainWithDeltaRi_TNR

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
    prob = -1
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    for temp in Array(inputDB_TestWithDeltaRi[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR = get1rate(inputDB_TestWithDeltaRi, prob)
            push!(TestWithDeltaRi_TPR, TPR)
            push!(TestWithDeltaRi_FNR, FNR)
            push!(TestWithDeltaRi_FDR, FDR)
            push!(TestWithDeltaRi_FPR, FPR)
            push!(TestWithDeltaRi_TNR, TNR)
        else
            push!(TestWithDeltaRi_TPR, TPR)
            push!(TestWithDeltaRi_FNR, FNR)
            push!(TestWithDeltaRi_FDR, FDR)
            push!(TestWithDeltaRi_FPR, FPR)
            push!(TestWithDeltaRi_TNR, TNR)
        end
    end
    inputDB_TestWithDeltaRi[!, "TPR"] = TestWithDeltaRi_TPR
    inputDB_TestWithDeltaRi[!, "FNR"] = TestWithDeltaRi_FNR
    inputDB_TestWithDeltaRi[!, "FDR"] = TestWithDeltaRi_FDR
    inputDB_TestWithDeltaRi[!, "FPR"] = TestWithDeltaRi_FPR
    inputDB_TestWithDeltaRi[!, "TNR"] = TestWithDeltaRi_TNR

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
    prob = -1
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    for temp in Array(inputDB_PestWithDeltaRi[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR = get1rate(inputDB_PestWithDeltaRi, prob)
            push!(PestWithDeltaRi_TPR, TPR)
            push!(PestWithDeltaRi_FNR, FNR)
            push!(PestWithDeltaRi_FDR, FDR)
            push!(PestWithDeltaRi_FPR, FPR)
            push!(PestWithDeltaRi_TNR, TNR)
        else
            push!(PestWithDeltaRi_TPR, TPR)
            push!(PestWithDeltaRi_FNR, FNR)
            push!(PestWithDeltaRi_FDR, FDR)
            push!(PestWithDeltaRi_FPR, FPR)
            push!(PestWithDeltaRi_TNR, TNR)
        end
    end
    inputDB_PestWithDeltaRi[!, "TPR"] = PestWithDeltaRi_TPR
    inputDB_PestWithDeltaRi[!, "FNR"] = PestWithDeltaRi_FNR
    inputDB_PestWithDeltaRi[!, "FDR"] = PestWithDeltaRi_FDR
    inputDB_PestWithDeltaRi[!, "FPR"] = PestWithDeltaRi_FPR
    inputDB_PestWithDeltaRi[!, "TNR"] = PestWithDeltaRi_TNR

## save ##, ouputing df 10908 x 23+5 df 
savePath = "F:\\UvA\\F\\UvA\\app\\dataframePostPredict_TPRFNRFDR_newPestNoTea_KNN.csv"
CSV.write(savePath, inputDB_PestWithDeltaRi)


# ==================================================================================================
## plot P(1)threshold-to-TPR & P(1)threshold-to-TNR ## for training set
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
TrainOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)
plot!(inputDB_TrainWithDeltaRi[:, end-6], [inputDB_TrainWithDeltaRi[:, end-4] inputDB_TrainWithDeltaRi[:, end-3]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=12, 
        label = ["True Positive Rate" "False Negative Rate"], 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = :left, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)
plot!(inputDB_TrainWithDeltaRi[:, end-6], inputDB_TrainWithDeltaRi[:, end-2], 
        subplot = 2, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=12, 
        label = "False Discovery Rate", 
        yguidefontsize=12, 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        ylims = [0, 0.23323328569055], 
        legend = :best, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)
        new_yticks = ([0.05], ["\$\\bar"], ["purple"])
        new_yticks2 = ([0.10], ["\$\\bar"], ["red"])
        hline!(new_yticks[1], label = "5% FDR-Controlled Cutoff at P(1) = 0.89", legendfont = font(10), lc = "purple", subplot = 2)
        hline!(new_yticks2[1], label = "10% FDR-Controlled Cutoff at P(1) = 0.78", legendfont = font(10), lc = "red", subplot = 2)
savefig(TrainOutplotP1toRate, "F:\\UvA\\F\\UvA\\app\\TPTNPrediction_P1threshold2TPRFNRFDR_newTrainALLylims_KNN.png")


# ==================================================================================================
## input ## 1686319 x 25 df; 421381 x 25 df; 10908 x 22 df; 8187 x 22 df
# columns: ENTRY, ID, INCHIKEY, INCHIKEYreal, 8 para, ISOTOPICMASS, 2 Ris, Delta Ri, LABEL, GROUP, Leverage, withDeltaRipredictTPTN, p0, p1
inputDB_TrainWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_TrainDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_TestWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_ValDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_PestWithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_PestDF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)
inputDB_Pest2WithDeltaRi = CSV.read("F:\\UvA\\F\\UvA\\app\\dataframeTPTNModeling_Pest2DF_withDeltaRIandPredictedTPTNandpTP_KNN.csv", DataFrame)


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
