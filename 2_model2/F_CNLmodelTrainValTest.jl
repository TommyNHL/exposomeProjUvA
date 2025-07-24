## INPUT(S)
# dataframe73_95dfTestSetWithStratification.csv
# dataframe73_95dfTrainSetWithStratification.csv
# CocamideExtWithStartification_Fingerprints_train.csv
# CocamideExtWithStratification_Fingerprints_test.csv
# dataAllFP_withNewPredictedRiWithStratification.csv

## OUTPUT(S)
# hyperparameterTuning_RFwithStratification10F.csv
# CocamideExtended73_CNLsRi_RFwithStratification.joblib
# dataframe73_dfTrainSetWithStratification_withCNLPredictedRi.csv
# dataframe73_dfTestSetWithStratification_withCNLPredictedRi.csv
# dataframe73_dfTrainSetWithStratification_withCNLPredictedRi_withCocamides.csv
# dataframe73_dfTestSetWithStratification_withCNLPredictedRi_withCocamides.csv
# CNLRiPrediction73_RFTrainWithStratification_v3.png
# CNLRiPrediction73_RFTestWithStratification_v3.png

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
#Conda.add("joblib")

## import packages from Python ##
jl = pyimport("joblib")
using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

## inputing 485577 x 15965 df for training ##
# columns: ENTRY, INCHIKEY, MONOISOTOPICMASS, CNLs, predictRi
#inputDB_test = CSV.read("F:\\UvA\\dataframe73_95dfTestSetWithStratification.csv", DataFrame)
inputDB_test = CSV.read("G:\\Temp\\dataframe73_95dfTestSetWithStratification.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])

## input 208104 x 15965 df for testing ##
#inputDB = CSV.read("F:\\UvA\\dataframe73_95dfTrainSetWithStratification.csv", DataFrame)
inputDB = CSV.read("G:\\Temp\\dataframe73_95dfTrainSetWithStratification.csv", DataFrame)
sort!(inputDB, [:ENTRY])

## copy ##
X = deepcopy(inputDB[:, 3:end-1])
size(X)
Y = deepcopy(inputDB[:, end])
size(Y)

## define a function for internal train/test split ##
function partitionTrainVal(df, ratio = 0.90)
    noOfRow = nrow(df)
    idx = shuffle(1:noOfRow)
    train_idx = view(idx, 1:floor(Int, ratio*noOfRow))
    test_idx = view(idx, (floor(Int, ratio*noOfRow)+1):noOfRow)
    df[train_idx,:], df[test_idx,:]
end

## define functions for performace evaluation
# Maximum absolute error
# mean square error (MSE) calculation
# Root mean square error (RMSE) calculation
function errorDetermination(arrRi, predictedRi)
    sumAE = 0
    sumRE = 0
    maxAE = 0
    for i = 1:size(predictedRi, 1)
        AE = abs(arrRi[i] - predictedRi[i])
        RE = (AE / predictedRi[i]) * 100
        if (AE > maxAE)
            maxAE = AE
        end
        sumAE += (AE ^ 2)
        sumRE += RE
    end
    MSE = sumAE / size(predictedRi, 1)
    RMSE = MSE ^ 0.5
    MRE = sumRE / size(predictedRi, 1)
    return maxAE, MSE, RMSE, MRE
end
#
# R-square value
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
#
## Average accuracy
function avgAcc(arrAcc, cv)
    sumAcc = 0
    for acc in arrAcc
        sumAcc += acc
    end
    return sumAcc / cv
end

## define a function for modeling ##
function optimRandomForestRegressor(df_train)
    #leaf_r = [collect(4:2:10);15;20]
    #leaf_r = vcat(collect(4:2:8), collect(12:4:20))
    #leaf_r = vcat(collect(1:1:2), collect(4:2:8))
    #leaf_r = collect(1:1:2)
    leaf_r = vcat(collect(1:1:2), collect(4))
    #tree_r = vcat(collect(50:50:400),collect(500:100:1000))
    #tree_r = collect(50:50:400)
    #tree_r = vcat(collect(50:50:400),collect(500:100:1000))
    tree_r = collect(50:50:400)
    rs = vcat(1, 42)
    z = zeros(1,7)
    itr = 1
    while itr < 9
        l = rand(leaf_r)
        t = rand(tree_r)
        for s in rs
            println("itr=", itr, ", leaf=", l, ", tree=", t, ", s=", s)
            MaxFeat = Int64(ceil((size(df_train,2)-1)/3))
            println("## split ##")
            M_train, M_val = partitionTrainVal(df_train, 0.90)
            Xx_train = deepcopy(M_train[:, 3:end-1])
            Yy_train = deepcopy(M_train[:, end])
            Xx_val = deepcopy(M_val[:, 3:end-1])
            Yy_val = deepcopy(M_val[:, end])
            println("## Regression ##")
            reg = RandomForestRegressor(n_estimators=t, min_samples_leaf=l, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=s)
            println("## fit ##")
            fit!(reg, Matrix(Xx_train), Vector(Yy_train))
            if itr == 1
                z[1,1] = l
                z[1,2] = t
                z[1,3] = score(reg, Matrix(Xx_train), Vector(Yy_train))
                z[1,4] = score(reg, Matrix(df_train[:, 3:end-1]), Vector(df_train[:, end]))
                println("## CV ##")
                acc10_train = cross_val_score(reg, Matrix(df_train[:, 3:end-1]), Vector(df_train[:, end]); cv = 10)
                z[1,5] = avgAcc(acc10_train, 10)
                z[1,6] = score(reg, Matrix(Xx_val), Vector(Yy_val))
                z[1,7] = s
                println(z[end, :])
            else
                println("## CV ##")
                itrain= score(reg, Matrix(Xx_train), Vector(Yy_train)) 
                traintrain = score(reg, Matrix(df_train[:, 3:end-1]), Vector(df_train[:, end]))
                acc10_train = cross_val_score(reg, Matrix(df_train[:, 3:end-1]), Vector(df_train[:, end]); cv = 10)
                traincvtrain = avgAcc(acc10_train, 10) 
                ival = score(reg, Matrix(Xx_val), Vector(Yy_val)) 
                z = vcat(z, [l t itrain traintrain traincvtrain ival s])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], accuracy_10Ftrain = z[:,3], accuracy_train = z[:,4], avgAccuracy10FCV_train = z[:,5], accuracy_val = z[:,6], state = z[:,7])
    z_df_sorted = sort(z_df, [:accuracy_val, :avgAccuracy10FCV_train, :accuracy_train, :accuracy_10Ftrain], rev=true)
    return z_df_sorted
end

## perform hyperparameter tuning ##
optiSearch_df = optimRandomForestRegressor(inputDB)

## save result ##
savePath = "F:\\UvA\\hyperparameterTuning_RFwithStratification10F.csv"
CSV.write(savePath, optiSearch_df)
# grid search
#= model = RandomForestRegressor()
param_dist = Dict(
      "n_estimators" => 50:50:300, 
      #"max_depth" => 2:2:10, 
      "min_samples_leaf" => 8:8:32, 
      "max_features" => [Int64(ceil((size(x_train,2)-1)/3))], 
      "n_jobs" => [-1], 
      "oob_score" => [true], 
      "random_state" => [1]
      )
gridsearch = GridSearchCV(model, param_dist)
@time fit!(gridsearch, Matrix(x_train), Vector(y_train))
println("Best parameters: $(gridsearch.best_params_)") =#

## build model ##
model = RandomForestRegressor(
      n_estimators = 350, 
      #max_depth = 10, 
      min_samples_leaf = 2, 
      max_features = Int64(ceil((size(inputDB[:, 3:end-1],2)-1)/3)), 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42
      )
fit!(model, Matrix(inputDB[:, 3:end-1]), Vector(inputDB[:, end]))

## save model ##
modelSavePath = "F:\\UvA\\CocamideExtended73_CNLsRi_RFwithStratification.joblib"
jl.dump(model, modelSavePath, compress = 5)


# ==================================================================================================
## evaluate performacnce ##
    # training performace, CNL-predictedRi vs. FP-predictedRi
    #model = jl.load("F:\\UvA\\CocamideExtended73_CNLsRi_RFwithStratification.joblib")
    model = jl.load("G:\\Temp\\CocamideExtended73_CNLsRi_RFwithStratification.joblib")
    predictedRi_train = predict(model, Matrix(inputDB[:, 3:end-1]))
    inputDB[!, "CNLpredictRi"] = predictedRi_train
    # save, ouputing trainSet df 0.7 x (3+15994+1)
    savePath = "F:\\UvA\\dataframe73_dfTrainSetWithStratification_withCNLPredictedRi.csv"
    CSV.write(savePath, inputDB)
    # call functions
    maxAE_train, MSE_train, RMSE_train, MRE_train = errorDetermination(inputDB[:, end-1], predictedRi_train)
    # 714.509, 2679.473, 51.764, 6.899%
    rSquare_train = rSquareDetermination(inputDB[:, end-1], predictedRi_train)
    # 0.9634
    # accuracy
    acc1_train = score(model, Matrix(inputDB[:, 3:end-2]), Vector(inputDB[:, end-1]))
    # 0.9634
    #acc5_train = cross_val_score(model, Matrix(inputDB[:, 3:end-2]), Vector(inputDB[:, end-1]); cv = 10)
    #avgAcc_train = avgAcc(acc5_train, 10)

    # testing performace, CNL-predictedRi vs. FP-predictedRi
    #modelRF_CNL = jl.load("F:\\UvA\\CocamideExtended73_CNLsRi_RFwithStratification.joblib")
    modelRF_CNL = jl.load("G:\\Temp\\CocamideExtended73_CNLsRi_RFwithStratification.joblib")
    size(modelRF_CNL)
    predictedRi_test = predict(modelRF_CNL, Matrix(inputDB_test[:, 3:end-1]))
    inputDB_test[!, "CNLpredictRi"] = predictedRi_test
    # save, ouputing testSet df 0.3 x (3+15994+1)
    savePath = "F:\\UvA\\dataframe73_dfTestSetWithStratification_withCNLPredictedRi.csv"
    CSV.write(savePath, inputDB_test)
    # call functions
    maxAE_val, MSE_val, RMSE_val, MRE_val = errorDetermination(inputDB_test[:, end-1], predictedRi_test)
    # 1057.714, 8613.806, 92.811, 9.915%
    rSquare_val = rSquareDetermination(inputDB_test[:, end-1], predictedRi_test)
    # 0.8824
    ## accuracy
    acc1_val = score(modelRF_CNL, Matrix(inputDB_test[:, 3:end-2]), Vector(inputDB_test[:, end-1]))
    # 0.8824
    #acc5_val = cross_val_score(modelRF_CNL, Matrix(inputDB_test[:, 3:end-2]), Vector(inputDB_test[:, end-1]); cv = 10)
    #avgAcc_val = avgAcc(acc5_val, 10)

## prepare to plot figures ##
    ## input dfs ## for separation of the cocamides and non-cocamides datasets
    # 5364 x 931 df 
    inputCocamidesTrain = CSV.read("G:\\Temp\\CocamideExtWithStartification_Fingerprints_train.csv", DataFrame)
    sort!(inputCocamidesTrain, :SMILES)
    # 947 x 931 df
    inputCocamidesTest = CSV.read("G:\\Temp\\CocamideExtWithStratification_Fingerprints_test.csv", DataFrame)
    sort!(inputCocamidesTest, :SMILES)
    # compare, 30684 x 793 df
    inputAllFPDB = CSV.read("G:\\Temp\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
    sort!(inputAllFPDB, [:INCHIKEY, :SMILES])
    #
function id2id(plotdf, i)
    inchikeyID = plotdf[i, "INCHIKEY"]
    idx = findall(inputAllFPDB.INCHIKEY .== inchikeyID)
    return inputAllFPDB[idx[end:end], "SMILES"][1]
end
#
function cocamidesOrNot(plotdf, i)
    if (id2id(plotdf, i) in Array(inputCocamidesTrain[:, "SMILES"]) || id2id(plotdf, i) in Array(inputCocamidesTest[:, "SMILES"]))
        return true
    else
        return false
    end
end
#
function findDots(plotdf, i)
    if (cocamidesOrNot(plotdf, i) == true)
        return plotdf[i, "predictRi"], plotdf[i, "CNLpredictRi"]
    end
end
#
trainCocamide = []
trainNonCocamide = []
inputDB[!, "Cocamides"] .= ""
for i in 1:size(inputDB, 1)
    if (cocamidesOrNot(inputDB, i) == true)
        inputDB[i, "Cocamides"] = "yes"
        push!(trainCocamide, i)
    elseif (cocamidesOrNot(inputDB, i) == false)
        inputDB[i, "Cocamides"] = "no"
        push!(trainNonCocamide, i)
    end
end
savePath = "F:\\UvA\\dataframe73_dfTrainSetWithStratification_withCNLPredictedRi_withCocamides.csv"
CSV.write(savePath, inputDB)
trainNonCocamide = findall(inputDB[:, "Cocamides"] .== "no")
trainCocamide = findall(inputDB[:, "Cocamides"] .== "yes")
#
testCocamide = []
testNonCocamide = []
inputDB_test[!, "Cocamides"] .= ""
for i in 1:size(inputDB_test, 1)
    if (cocamidesOrNot(inputDB_test, i) == true)
        inputDB_test[i, "Cocamides"] = "yes"
        push!(testCocamide, i)
    elseif (cocamidesOrNot(inputDB_test, i) == false)
        inputDB_test[i, "Cocamides"] = "no"
        push!(testNonCocamide, i)
    end
end
savePath = "F:\\UvA\\dataframe73_dfTestSetWithStratification_withCNLPredictedRi_withCocamides.csv"
CSV.write(savePath, inputDB_test)
testNonCocamide = findall(inputDB_test[:, "Cocamides"] .== "no")
testCocamide = findall(inputDB_test[:, "Cocamides"] .== "yes")

## plot figures ##
# inputing 693685*0.3 x 1+1+1+15961+1+1+1 df = 208106 x 15965
# columns: ENTRY, INCHIKEY, ISOTOPICMASS, CNLs, predictRi, CNLpredictRi, Cocamides
inputDB_test = CSV.read("G:\\Temp\\dataframe73_dfTestSetWithStratification_withCNLPredictedRi_withCocamides.csv", DataFrame)
# inputing 693685*0.7 x 1+1+1+15961+1 df = 485579 x 15965
inputDB = CSV.read("G:\\Temp\\dataframe73_dfTrainSetWithStratification_withCNLPredictedRi_withCocamides.csv", DataFrame)
#
inputDB_test[:, end-2:end]
inputDB[:, end-2:end]
#
#Pkg.add("Distributions")
using Distributions
    ## plot for training ##
    tempDfTrain = inputDB[:, end-1]
    layout = @layout [a{0.8w,0.2h}            _
                    b{0.8w,0.8h} c{0.2w,0.8h}]
    describe(inputDB[:, "CNLpredictRi"])
    default(fillcolor = :lightgrey, grid = false, legend = false)
    outplotTrain = plot(layout = layout, link = :both, legend = :topleft, 
            size = (600, 600), margin = -2Plots.px, dpi = 300)
    scatter!(inputDB[trainNonCocamide, end-2], inputDB[trainNonCocamide, end-1], 
            subplot = 2, framestyle = :box, 
            xlabel = "MF-derived RTI values", ylabel = "CNL-derived RTI values", 
            markershape = :star, 
            c = :skyblue, 
            markerstrokewidth = 0, 
            alpha = 0.50, 
            label = "Extended", 
            margin = -2Plots.px, 
            size = (600,600), 
            dpi = 300)
    scatter!(inputDB[trainCocamide, end-2], inputDB[trainCocamide, end-1], 
            subplot = 2, framestyle = :box, 
            xlabel = "MF-derived RTI values", ylabel = "CNL-derived RTI values", 
            markershape = :star, 
            c = :green, 
            markerstrokewidth = 0, 
            alpha = 0.25, 
            label = "Compound Calibrants", 
            margin = -2Plots.px, 
            size = (600,600), 
            dpi = 300)
    plot!(tempDfTrain -> tempDfTrain, c=:red, subplot = 2, 
            label = "Identity Line", 
            margin = -2Plots.px, 
            size = (600,600), 
            dpi = 300)
    histogram!(inputDB[:, end-2], bins = 165, subplot = 1, 
            xlims = (-150, 1500), 
            label = false, 
            orientation = :v, 
            framestyle = :none, 
            dpi = 300)
    histogram!(inputDB[:, end-1], bins = 165, subplot = 3, 
            ylims = (-150, 1500), 
            label = false, 
            orientation = :h, 
            framestyle = :none, 
            dpi = 300)
    plot!(xlims = (-150, 1500), ylims = (-150, 1500), subplot = 2)
## save ##
savefig(outplotTrain, "G:\\Temp\\CNLRiPrediction73_RFTrainWithStratification_v4.png")
#
    ## plot for testing ##
    tempDfTest = inputDB_test[:, end-1]
    layout = @layout [a{0.8w,0.2h}            _
                    b{0.8w,0.8h} c{0.2w,0.8h}]
    describe(inputDB_test[:, "CNLpredictRi"])
    default(fillcolor = :lightgrey, grid = false, legend = false)
    outplotTest = plot(layout = layout, link = :both, legend = :topleft, 
            size = (600, 600), margin = -2Plots.px, dpi = 300)
    scatter!(inputDB_test[testNonCocamide, end-2], inputDB_test[testNonCocamide, end-1], 
            subplot = 2, framestyle = :box, 
            xlabel = "MF-derived RTI values", ylabel = "CNL-derived RTI values", 
            markershape = :star, 
            c = :skyblue, 
            markerstrokewidth = 0, 
            alpha = 0.50, 
            label = "Extended", 
            margin = -2Plots.px, 
            size = (600,600), 
            dpi = 300)
    scatter!(inputDB_test[testCocamide, end-2], inputDB_test[testCocamide, end-1], 
            subplot = 2, framestyle = :box, 
            xlabel = "MF-derived RTI values", ylabel = "CNL-derived RTI values", 
            markershape = :star, 
            c = :green, 
            markerstrokewidth = 0, 
            alpha = 0.25, 
            label = "Compound Calibrants", 
            margin = -2Plots.px, 
            size = (600,600), 
            dpi = 300)
    plot!(tempDfTest -> tempDfTest, c=:red, subplot = 2, 
            label = "Identity Line", 
            margin = -2Plots.px, 
            size = (600,600), 
            dpi = 300)
    histogram!(inputDB_test[:, end-2], bins = 165, subplot = 1, 
            xlims = (-150, 1500), 
            label = false, 
            orientation = :v, 
            framestyle = :none, 
            dpi = 300)
    histogram!(inputDB_test[:, end-1], bins = 165, subplot = 3, 
            ylims = (-150, 1500), 
            label = false, 
            orientation = :h, 
            framestyle = :none, 
            dpi = 300)
    plot!(xlims = (-150, 1500), ylims = (-150, 1500), subplot = 2)
## save ##
savefig(outplotTest, "G:\\Temp\\CNLRiPrediction73_RFTestWithStratification_v4.png")
