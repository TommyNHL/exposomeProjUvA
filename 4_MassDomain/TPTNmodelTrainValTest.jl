VERSION
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add("Plots")
#Pkg.add("ProgressBars")
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
#Pkg.add(PackageSpec(url=""))
using Random
#using BSON
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots
#using ProgressBars
#using PyPlot
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#using PyCall, Conda                 #using python packages
#pcp = pyimport("pubchempy")
pd = pyimport("padelpy")            #calculation of FP
jl = pyimport("joblib")             # used for loading models
f1_score = pyimport("sklearn.metrics").f1_score
matthews_corrcoef = pyimport("sklearn.metrics").matthews_corrcoef

using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
@sk_import metrics: recall_score
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

# inputing 827145 x 4+9+2+1+2 df
# columns: ENTRY, INCHIKEY, ISOTOPICMASS, CNLs, predictRi
inputDB_test = CSV.read("F:\\dataframeTPTNModeling_TestDF.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
# inputing 3308576 x 4+9+2+1+2 df
inputDB = CSV.read("F:\\dataframeTPTNModeling_TrainDF.csv", DataFrame)
sort!(inputDB, [:ENTRY])



function partitionTrainVal(df, ratio = 0.67)
    noOfRow = nrow(df)
    idx = shuffle(1:noOfRow)
    train_idx = view(idx, 1:floor(Int, ratio*noOfRow))
    test_idx = view(idx, (floor(Int, ratio*noOfRow)+1):noOfRow)
    df[train_idx,:], df[test_idx,:]
end

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

# modeling, 9 x 11 = 99 times
function optimRandomForestRegressor(df_train)
    #leaf_r = [collect(4:2:10);15;20]
    leaf_r = vcat(collect(1:1:9))
    tree_r = vcat(collect(50:50:400),collect(500:100:700))
    #tree_r = collect(50:50:300)
    z = zeros(1,8)
    itr = 1
    while itr < 66
        l = rand(leaf_r)
        t = rand(tree_r)
        println("itr=", itr, ", leaf=", l, ", tree=", t)
        MaxFeat = Int64(9)
        println("## split ##")
        M_train, M_val = partitionTrainVal(df_train, 0.67)
        Xx_train = deepcopy(M_train[:, vcat(collect(5:12), end-3)])
        Yy_train = deepcopy(M_train[:, end-2])
        Xx_val = deepcopy(M_val[:, vcat(collect(5:12), end-3)])
        Yy_val = deepcopy(M_val[:, end-2])
        println("## Regression ##")
        reg = RandomForestClassifier(n_estimators=t, min_samples_leaf=l, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=42, class_weight=Dict(0=>0.529, 1=>9.021))
        println("## fit ##")
        fit!(reg, Matrix(Xx_train), Vector(Yy_train))
        if itr == 1
            z[1,1] = l
            z[1,2] = t
            z[1,3] = score(reg, Matrix(Xx_train), Vector(Yy_train))
            z[1,4] = score(reg, Matrix(df_train[:, vcat(collect(5:12), end-3)]), Vector(df_train[:, end-2]))
            println("## CV ##")
            acc5_train = cross_val_score(reg, Matrix(df_train[:, vcat(collect(5:12), end-3)]), Vector(df_train[:, end-2]); cv = 3)
            z[1,5] = avgAcc(acc5_train, 3)
            z[1,6] = score(reg, Matrix(Xx_val), Vector(Yy_val))
            z[1,7] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
            z[1,8] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
            println(z[end, :])
        else
            println("## CV ##")
            itrain= score(reg, Matrix(Xx_train), Vector(Yy_train)) 
            traintrain = score(reg, Matrix(df_train[:, vcat(collect(5:12), end-3)]), Vector(df_train[:, end-2]))
            acc5_train = cross_val_score(reg, Matrix(df_train[:, vcat(collect(5:12), end-3)]), Vector(df_train[:, end-2]); cv = 3)
            traincvtrain = avgAcc(acc5_train, 3) 
            ival = score(reg, Matrix(Xx_val), Vector(Yy_val))
            f1s = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
            mccs = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)))
            z = vcat(z, [l t itrain traintrain traincvtrain ival f1s mccs])
            println(z[end, :])
        end
        println("End of ", itr, " iterations")
        itr += 1
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], accuracy_3Ftrain = z[:,3], accuracy_train = z[:,4], avgAccuracy3FCV_train = z[:,5], accuracy_val = z[:,6], f1_val = z[:,7], mcc_val = z[:,8])
    z_df_sorted = sort(z_df, [:mcc_val, :f1_val, :accuracy_val, :avgAccuracy3FCV_train, :accuracy_train, :accuracy_3Ftrain], rev=true)
    return z_df_sorted
end

optiSearch_df = optimRandomForestRegressor(inputDB)

# save, ouputing 180 x 8 df
savePath = "F:\\hyperparameterTuning_TPTNwithDeltaRi.csv"
CSV.write(savePath, optiSearch_df)

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

model = RandomForestClassifier(
      n_estimators = 250, 
      #max_depth = 10, 
      min_samples_leaf = 2, 
      max_features = Int64(9), 
      n_jobs = -1, 
      oob_score = true, 
      random_state = 42, 
      class_weight= Dict(0=>0.529, 1=>9.021)
      )
#fit!(model, Matrix(inputDB[:, vcat(collect(5:12), end-3)]), Vector(inputDB[:, end-2]))
fit!(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-2]))

# saving model
#modelSavePath = "J:\\UvA\\1_model\\modelTPTNModeling_withDeltaRi.joblib"
modelSavePath = "J:\\UvA\\1_model\\modelTPTNModeling_withoutDeltaRi.joblib"
jl.dump(model, modelSavePath, compress = 5)

# training performace, withDeltaRi vs. withoutDeltaRi
#= predictedTPTN_train = predict(model,  Matrix(inputDB[:, vcat(collect(5:12), end-3)]))
inputDB[!, "withDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3308576 x 19 df
savePath = "F:\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB) =#
# --------------------------------------------------------------------------------------------------
predictedTPTN_train = predict(model, Matrix(inputDB[:, 5:12]))
inputDB[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_train
# save, ouputing trainSet df 3308576 x 19 df
savePath = "F:\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB)

# ==================================================================================================

#= # 1.0, 0.053116907732042806, 0.23047105616984273
maxAE_train, MSE_train, RMSE_train = errorDetermination(Matrix(inputDB[:, vcat(collect(5:12), end-4)]), predictedTPTN_train)
# -3.2123231824226766
rSquare_train = rSquareDetermination(Matrix(inputDB[:, vcat(collect(5:12), end-4)]), predictedTPTN_train)
## accuracy, 0.9977256076330119
acc1_train = score(model, Matrix(inputDB[:, vcat(collect(5:12), end-4)]), Vector(inputDB[:, end-3]))
# 0.9375740688519566, 0.9334203193699285, 0.9360479771647846
acc5_train = cross_val_score(model, Matrix(inputDB[:, vcat(collect(5:12), end-4)]), Vector(inputDB[:, end-3]); cv = 3)
# 0.9356807884622232
avgAcc_train = avgAcc(acc5_train, 3)
# 3308576 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB[:, vcat(collect(5:12), end-4)]))
# 0.9798879068188683
f1_train = f1_score(Vector(inputDB[:, end-3]), predictedTPTN_train)
# 0.978889452177055
mcc_train = matthews_corrcoef(Vector(inputDB[:, end-3]), predictedTPTN_train)

inputDB[!, "pTP_train1"] = pTP_train[:, 1]
inputDB[!, "pTP_train2"] = pTP_train[:, 2]
# save, ouputing trainSet df 3308576 x (18+1+2)
savePath = "F:\\dataframeTPTNModeling_TrainDF_withDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB)

describe((inputDB))[end-4:end, :] =#
# --------------------------------------------------------------------------------------------------
# 1.0, 0.10199372858240166, 0.3193645700174045
maxAE_train, MSE_train, RMSE_train = errorDetermination(Matrix(inputDB[:, 5:12]), predictedTPTN_train)
# -6.375214361197565
rSquare_train = rSquareDetermination(Matrix(inputDB[:, 5:12]), predictedTPTN_train)
## accuracy, 0.9476439410791833
acc1_train = score(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-3]))
#  0.9088469151541584, 0.8994622159315017, 0.9046622502624998
acc5_train = cross_val_score(model, Matrix(inputDB[:, 5:12]), Vector(inputDB[:, end-3]); cv = 3)
# 0.9043237937827199
avgAcc_train = avgAcc(acc5_train, 3)
# 3308576 × 2 Matrix
pTP_train = predict_proba(model, Matrix(inputDB[:, 5:12]))
# 0.6791684724830624
f1_train = f1_score(Vector(inputDB[:, end-3]), predictedTPTN_train)
# 0.6968009156974482
mcc_train = matthews_corrcoef(Vector(inputDB[:, end-3]), predictedTPTN_train)

inputDB[!, "pTP_train1"] = pTP_train[:, 1]
inputDB[!, "pTP_train2"] = pTP_train[:, 2]
# save, ouputing trainSet df 3308576 x (18+1+2)
savePath = "F:\\dataframeTPTNModeling_TrainDF_withoutDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB)

describe((inputDB))[end-4:end, :]

# ==================================================================================================

# model validation
# load a model
# requires python 3.11 or 3.12
modelRF_TPTN = jl.load("J:\\UvA\\1_model\\modelTPTNModeling_withDeltaRi.joblib")
size(modelRF_TPTN)
# --------------------------------------------------------------------------------------------------
modelRF_TPTN = jl.load("J:\\UvA\\1_model\\modelTPTNModeling_withoutDeltaRi.joblib")
size(modelRF_TPTN)

# ==================================================================================================

predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-3)]))
inputDB_test[!, "withDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 827145 x 19 df
savePath = "F:\\dataframeTPTNModeling_TestDF_withDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB_test)
# --------------------------------------------------------------------------------------------------
predictedTPTN_test = predict(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]))
inputDB_test[!, "withoutDeltaRipredictTPTN"] = predictedTPTN_test
# save, ouputing testSet df 827145 x 19 df
savePath = "F:\\dataframeTPTNModeling_TestDF_withoutDeltaRiandPredictedTPTN.csv"
CSV.write(savePath, inputDB_test)

# ==================================================================================================
# 1.0, 0.050349413021199635, 0.22438674876471568
maxAE_val, MSE_val, RMSE_val = errorDetermination(Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]), predictedTPTN_test)
# -2.973935605896406
rSquare_val = rSquareDetermination(Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]), predictedTPTN_test)
## accuracy, 0.983563945861971
acc1_val = score(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]), Vector(inputDB_test[:, end-3]))
#  0.9387846145476307, 0.9334203797399488, 0.9374245144442631
acc5_val = cross_val_score(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]), Vector(inputDB_test[:, end-3]); cv = 3)
# 0.9365431695772809
avgAcc_val = avgAcc(acc5_val, 3)
# 827145 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputDB_test[:, vcat(collect(5:12), end-4)]))
# 0.8499895174727179
f1_test = f1_score(Vector(inputDB_test[:, end-3]), predictedTPTN_test)
# 0.8413454155937011
mcc_test = matthews_corrcoef(Vector(inputDB_test[:, end-3]), predictedTPTN_test)

inputDB_test[!, "pTP_test1"] = pTP_test[:, 1]
inputDB_test[!, "pTP_test2"] = pTP_test[:, 2]
# save, ouputing trainSet df 0.7 x (3+15994+1)
savePath = "F:\\dataframeTPTNModeling_TestDF_withDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB_test)

describe((inputDB_test))[end-4:end, :]
# --------------------------------------------------------------------------------------------------
# 1.0, 0.09573077136546966, 0.3094038968168786
maxAE_val, MSE_val, RMSE_val = errorDetermination(Matrix(inputDB_test[:, 5:12]), predictedTPTN_test)
# -6.206275194285345
rSquare_val = rSquareDetermination(Matrix(inputDB_test[:, 5:12]), predictedTPTN_test)
## accuracy, 0.91616584758416
acc1_val = score(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]), Vector(inputDB_test[:, end-3]))
#  0.9095406488584227, 0.899718912645304, 0.9064940246268792
acc5_val = cross_val_score(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]), Vector(inputDB_test[:, end-3]); cv = 3)
# 0.9052511953768686
avgAcc_val = avgAcc(acc5_val, 3)
# 827145 × 2 Matrix
pTP_test = predict_proba(modelRF_TPTN, Matrix(inputDB_test[:, 5:12]))
# 0.4613424685201153
f1_test = f1_score(Vector(inputDB_test[:, end-3]), predictedTPTN_test)
# 0.4418821354064598
mcc_test = matthews_corrcoef(Vector(inputDB_test[:, end-3]), predictedTPTN_test)

inputDB_test[!, "pTP_test1"] = pTP_test[:, 1]
inputDB_test[!, "pTP_test2"] = pTP_test[:, 2]
# save, ouputing trainSet df 827145 x 19+2 df 
savePath = "F:\\dataframeTPTNModeling_TestDF_withoutDeltaRiandPredictedTPTNandpTP.csv"
CSV.write(savePath, inputDB_test)

describe((inputDB_test))[end-4:end, :]

# ==================================================================================================

# plots
# inputing dfs for separation of the cocamides and non-cocamides datasets
## 5364 x 931 df 
inputCocamidesTrain = CSV.read("F:\\CocamideExtWithStartification_Fingerprints_train.csv", DataFrame)
sort!(inputCocamidesTrain, :SMILES)

## 947 x 931 df
inputCocamidesTest = CSV.read("F:\\CocamideExtWithStratification_Fingerprints_test.csv", DataFrame)
sort!(inputCocamidesTest, :SMILES)

# comparing, 30684 x 793 df
inputAllFPDB = CSV.read("F:\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
sort!(inputAllFPDB, [:INCHIKEY, :SMILES])

function id2id(plotdf, i)
    inchikeyID = plotdf[i, "INCHIKEY"]
    idx = findall(inputAllFPDB.INCHIKEY .== inchikeyID)
    return inputAllFPDB[idx[end:end], "SMILES"][1]
end

function cocamidesOrNot(plotdf, i)
    if (id2id(plotdf, i) in Array(inputCocamidesTrain[:, "SMILES"]) || id2id(plotdf, i) in Array(inputCocamidesTest[:, "SMILES"]))
        return true
    else
        return false
    end
end

function findDots(plotdf, i)
    if (cocamidesOrNot(plotdf, i) == true)
        return plotdf[i, "predictRi"], plotdf[i, "CNLpredictRi"]
    end
end

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
savePath = "F:\\dataframe_dfTrainSetWithStratification_withCNLPredictedRi_withCocamides.csv"
CSV.write(savePath, inputDB)
trainNonCocamide = findall(inputDB[:, "Cocamides"] .== "no")
trainCocamide = findall(inputDB[:, "Cocamides"] .== "yes")

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
savePath = "F:\\dataframe_dfTestSetWithStratification_withCNLPredictedRi_withCocamides.csv"
CSV.write(savePath, inputDB_test)
testNonCocamide = findall(inputDB_test[:, "Cocamides"] .== "no")
testCocamide = findall(inputDB_test[:, "Cocamides"] .== "yes")

# inputing 693685*0.3 x 1+1+1+15961+1+1+1 df = 208106 x 15965
# columns: ENTRY, INCHIKEY, ISOTOPICMASS, CNLs, predictRi, CNLpredictRi, Cocamides
inputDB_test = CSV.read("F:\\dataframe_dfTestSetWithStratification_withCNLPredictedRi_withCocamides.csv", DataFrame)

# inputing 693685*0.7 x 1+1+1+15961+1 df = 485579 x 15965
inputDB = CSV.read("F:\\dataframe_dfTrainSetWithStratification_withCNLPredictedRi_withCocamides.csv", DataFrame)

#Pkg.add("Distributions")
using Distributions
tempDfTrain = inputDB[:, end-1]
layout = @layout [a{0.8w,0.2h}            _
                  b{0.8w,0.8h} c{0.2w,0.8h}]
describe(inputDB[:, "CNLpredictRi"])
default(fillcolor = :lightgrey, grid = false, legend = false)
outplotTrain = plot(layout = layout, link = :both, 
        size = (600, 600), margin = -2Plots.px, dpi = 300)
scatter!(inputDB[trainNonCocamide, end-2], inputDB[trainNonCocamide, end-1], 
        subplot = 2, framestyle = :box, 
        xlabel = "FP-derived Ri values", ylabel = "CNL-derived Ri values", 
        markershape = :star, 
        c = :skyblue, 
        markerstrokewidth = 0, 
        alpha = 0.1, 
        label = "Non-Cocamides", 
        margin = -2Plots.px, 
        size = (600,600), 
        dpi = 300)
scatter!(inputDB[trainCocamide, end-2], inputDB[trainCocamide, end-1], 
        subplot = 2, framestyle = :box, 
        xlabel = "FP-derived Ri values", ylabel = "CNL-derived Ri values", 
        markershape = :star, 
        c = :green, 
        markerstrokewidth = 0, 
        alpha = 0.1, 
        label = "Cocamides", 
        margin = -2Plots.px, 
        size = (600,600), 
        dpi = 300)
plot!(tempDfTrain -> tempDfTrain, c=:red, subplot = 2, 
        label = false, 
        margin = -2Plots.px, 
        size = (600,600), 
        dpi = 300)
histogram!(inputDB[:, end-2], subplot = 1, 
        xlims = (-150, 1500), 
        orientation = :v, 
        framestyle = :none, 
        dpi = 300)
histogram!(inputDB[:, end-1], subplot = 3, 
        ylims = (-150, 1500), 
        orientation = :h, 
        framestyle = :none, 
        dpi = 300)
plot!(xlims = (-150, 1500), ylims = (-150, 1500), subplot = 2)
        # Saving
savefig(outplotTrain, "F:\\CNLRiPrediction_RFTrainWithStratification_v2.png")

tempDfTest = inputDB_test[:, end-1]
layout = @layout [a{0.8w,0.2h}            _
                  b{0.8w,0.8h} c{0.2w,0.8h}]
describe(inputDB_test[:, "CNLpredictRi"])
default(fillcolor = :lightgrey, grid = false, legend = false)
outplotTest = plot(layout = layout, link = :both, 
        size = (600, 600), margin = -2Plots.px, dpi = 300)
scatter!(inputDB_test[testNonCocamide, end-2], inputDB_test[testNonCocamide, end-1], 
        subplot = 2, framestyle = :box, 
        xlabel = "FP-derived Ri values", ylabel = "CNL-derived Ri values", 
        markershape = :star, 
        c = :skyblue, 
        markerstrokewidth = 0, 
        alpha = 0.1, 
        label = "Non-Cocamides", 
        margin = -2Plots.px, 
        size = (600,600), 
        dpi = 300)
scatter!(inputDB_test[testCocamide, end-2], inputDB_test[testCocamide, end-1], 
        subplot = 2, framestyle = :box, 
        xlabel = "FP-derived Ri values", ylabel = "CNL-derived Ri values", 
        markershape = :star, 
        c = :green, 
        markerstrokewidth = 0, 
        alpha = 0.1, 
        label = "Cocamides", 
        margin = -2Plots.px, 
        size = (600,600), 
        dpi = 300)
plot!(tempDfTest -> tempDfTest, c=:red, subplot = 2, 
        label = false, 
        margin = -2Plots.px, 
        size = (600,600), 
        dpi = 300)
histogram!(inputDB_test[:, end-2], subplot = 1, 
        xlims = (-150, 1500), 
        orientation = :v, 
        framestyle = :none, 
        dpi = 300)
histogram!(inputDB_test[:, end-1], subplot = 3, 
        ylims = (-150, 1500), 
        orientation = :h, 
        framestyle = :none, 
        dpi = 300)
plot!(xlims = (-150, 1500), ylims = (-150, 1500), subplot = 2)
        # Saving
savefig(outplotTest, "F:\\CNLRiPrediction_RFTestWithStratification_v2.png")