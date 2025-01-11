VERSION
## install packages needed ##
using Pkg
#Pkg.add("PyCall")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall

## import packages from Python ##
#Conda.add("joblib")
jl = pyimport("joblib")
using ScikitLearn: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier

## define a function for condensing PubChem FingerPrinter features ##
function convertPubChemFPs(ACfp::DataFrame, PCfp::DataFrame)
    FP1tr = ACfp
    pubinfo = Matrix(PCfp)
    # ring counts
    FP1tr[!,"PCFP-r3"] = pubinfo[:,1]
    FP1tr[!,"PCFP-r3"][pubinfo[:,8] .== 1] .= 2
    FP1tr[!,"PCFP-r4"] = pubinfo[:,15]
    FP1tr[!,"PCFP-r4"][pubinfo[:,22] .== 1] .= 2
    FP1tr[!,"PCFP-r5"] = pubinfo[:,29]
    FP1tr[!,"PCFP-r5"][pubinfo[:,36] .== 1] .= 2
    FP1tr[!,"PCFP-r5"][pubinfo[:,43] .== 1] .= 3
    FP1tr[!,"PCFP-r5"][pubinfo[:,50] .== 1] .= 4
    FP1tr[!,"PCFP-r5"][pubinfo[:,57] .== 1] .= 5
    # cont.
    FP1tr[!,"PCFP-r6"] = pubinfo[:,64]
    FP1tr[!,"PCFP-r6"][pubinfo[:,71] .== 1] .= 2
    FP1tr[!,"PCFP-r6"][pubinfo[:,78] .== 1] .= 3
    FP1tr[!,"PCFP-r6"][pubinfo[:,85] .== 1] .= 4
    FP1tr[!,"PCFP-r6"][pubinfo[:,92] .== 1] .= 5
    FP1tr[!,"PCFP-r7"] = pubinfo[:,99]
    FP1tr[!,"PCFP-r7"][pubinfo[:,106] .== 1] .= 2
    FP1tr[!,"PCFP-r8"] = pubinfo[:,113]
    FP1tr[!,"PCFP-r8"][pubinfo[:,120] .== 1] .= 2
    FP1tr[!,"PCFP-r9"] = pubinfo[:,127]
    FP1tr[!,"PCFP-r10"] = pubinfo[:,134]
    # minimum number of type of rings
    arom = zeros(size(pubinfo,1))
    arom[(arom .== 0) .& (pubinfo[:,147] .== 1)] .= 4
    arom[(arom .== 0) .& (pubinfo[:,145] .== 1)] .= 3
    arom[(arom .== 0) .& (pubinfo[:,143] .== 1)] .= 2
    arom[(arom .== 0) .& (pubinfo[:,141] .== 1)] .= 1
    FP1tr[!,"minAromCount"] = arom
    het = zeros(size(pubinfo,1))
    het[(het .== 0) .& (pubinfo[:,148] .== 1)] .= 4
    het[(het .== 0) .& (pubinfo[:,146] .== 1)] .= 3
    het[(het .== 0) .& (pubinfo[:,144] .== 1)] .= 2
    het[(het .== 0) .& (pubinfo[:,142] .== 1)] .= 1
    FP1tr[!,"minHetrCount"] = het
    # cont.
    return FP1tr
end

## predict Model 1 training set
    ## import csv ##
    # 5048 x 932 df 
    inputCocamidesTrain = CSV.read("G:\\Temp\\CocamideExtWithStartification_Fingerprints_train.csv", DataFrame)

    # call function to convert PubChem FingerPrinter features ##
    inputCocamidesTrain = convertPubChemFPs(inputCocamidesTrain[:,1:5+779], inputCocamidesTrain[:,5+780:end])
   
    # replace missing values ##
    ind_ap2d = vec(any(ismissing.(Matrix(inputCocamidesTrain[:, 5:end-10])) .== 1, dims = 2))
    ind_pub = vec(any(ismissing.(Matrix(inputCocamidesTrain[:, end-9:end])) .== 1, dims = 2))
    ind_all = ind_ap2d .+ ind_pub
    input_ap2d = inputCocamidesTrain[:, 5:end-10][ind_all .== 0, :]
    input_pub = inputCocamidesTrain[:, end-9:end][ind_all .== 0, :]
    allFPs_train = hcat(input_ap2d, input_pub)

    ## load a model ##
    # RandomForestRegressor(max_features=310, min_samples_leaf=8, n_estimators=200), size 200
    # requires python 3.11 or 3.12
    modelRF = jl.load("G:\\Temp\\CocamideExtendedWithStratification.joblib")
    size(modelRF)
        
    ## apply model ##
    # requires sklearn v1.3.1 has been installed on Python 3.11 environment
    predictedRi_train = predict(modelRF, Matrix(allFPs_train))
    inputCocamidesTrain[!, "predictRi"] = predictedRi_train

    ## save the output table as a spreadsheet ##
    savePath = "G:\\Temp\\CocamideExtWithStartification_Fingerprints_train_predict.csv"
    CSV.write(savePath, inputCocamidesTrain)

    inputCocamidesTrain = inputCocamidesTrain[:, vcat(1,3, end)]
    sort!(inputCocamidesTrain, [:SMILES])
    inputCocamidesTrain[!, "dataset"] .= "train"
    inputCocamidesTrain[!, "Error"] .= float(0)
    inputCocamidesTrain[!, "AbsError"] .= float(0)
    for i in 1:size(inputCocamidesTrain, 1)
        inputCocamidesTrain[i, "Error"] = inputCocamidesTrain[i, "predictRi"] - inputCocamidesTrain[i, "RI"]
        inputCocamidesTrain[i, "AbsError"] = abs(inputCocamidesTrain[i, "Error"])
    end
    sort!(inputCocamidesTrain, [:AbsError])
    print(inputCocamidesTrain[1, "AbsError"])  # 0.06, 0.05652
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.25), "AbsError"])  # 30.34
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.5), "AbsError"])  # 65.89
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.75), "AbsError"])  # 121.36
    print(inputCocamidesTrain[size(inputCocamidesTrain, 1)*1, "AbsError"])  # 1274.32
    describe(inputCocamidesTrain)
    # avgAbsError = 89.52

    sort!(inputCocamidesTrain, [:Error])
    print(inputCocamidesTrain[1, "Error"])  # -1274.32
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.25), "Error"])  # -61.21
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.5), "Error"])  # 11.16
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.75), "Error"])  # 68.63
    print(inputCocamidesTrain[size(inputCocamidesTrain, 1)*1, "Error"])  # 569.29
    describe(inputCocamidesTrain)
    # avgError = 0.20

    ## save the output table as a spreadsheet ##
    savePath = "G:\\Temp\\CocamideExtWithStartification_Fingerprints_train_predict_err.csv"
    CSV.write(savePath, inputCocamidesTrain)

## predict Model 1 testing set
    ## import csv ##
    # 5048 x 932 df 
    inputCocamidesTest = CSV.read("G:\\Temp\\CocamideExtWithStratification_Fingerprints_test.csv", DataFrame)

    # call function to convert PubChem FingerPrinter features ##
    inputCocamidesTest = convertPubChemFPs(inputCocamidesTest[:,1:5+779], inputCocamidesTest[:,5+780:end])
   
    # replace missing values ##
    ind_ap2d = vec(any(ismissing.(Matrix(inputCocamidesTest[:, 5:end-10])) .== 1, dims = 2))
    ind_pub = vec(any(ismissing.(Matrix(inputCocamidesTest[:, end-9:end])) .== 1, dims = 2))
    ind_all = ind_ap2d .+ ind_pub
    input_ap2d = inputCocamidesTest[:, 5:end-10][ind_all .== 0, :]
    input_pub = inputCocamidesTest[:, end-9:end][ind_all .== 0, :]
    allFPs_test = hcat(input_ap2d, input_pub)
        
    ## apply model ##
    # requires sklearn v1.3.1 has been installed on Python 3.11 environment
    predictedRi_test = predict(modelRF, Matrix(allFPs_test))
    inputCocamidesTest[!, "predictRi"] = predictedRi_test

    ## save the output table as a spreadsheet ##
    savePath = "G:\\Temp\\CocamideExtWithStartification_Fingerprints_test_predict.csv"
    CSV.write(savePath, inputCocamidesTest)

    inputCocamidesTest = inputCocamidesTest[:, vcat(1,3, end)]
    sort!(inputCocamidesTest, [:SMILES])
    inputCocamidesTest[!, "dataset"] .= "test"
    inputCocamidesTest[!, "Error"] .= float(0)
    inputCocamidesTest[!, "AbsError"] .= float(0)
    for i in 1:size(inputCocamidesTest, 1)
        inputCocamidesTest[i, "Error"] = inputCocamidesTest[i, "predictRi"] - inputCocamidesTest[i, "RI"]
        inputCocamidesTest[i, "AbsError"] = abs(inputCocamidesTest[i, "Error"])
    end
    sort!(inputCocamidesTest, [:AbsError])
    print(inputCocamidesTest[1, "AbsError"])  # 0.01, 0.007127
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.25), "AbsError"])  # 38.57
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.5), "AbsError"])  # 82.94
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.75), "AbsError"])  # 154.15
    print(inputCocamidesTest[size(inputCocamidesTest, 1)*1, "AbsError"])  # 953.91
    describe(inputCocamidesTest)
    # avgAbsError = 111.15

    sort!(inputCocamidesTest, [:Error])
    print(inputCocamidesTest[1, "Error"])  # -953.91
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.25), "Error"])  # -81.94
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.5), "Error"])  # 6.47
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.75), "Error"])  # 83.17
    print(inputCocamidesTest[size(inputCocamidesTest, 1)*1, "Error"])  # 599.52
    describe(inputCocamidesTest)
    # avgError = -1.54

    ## save the output table as a spreadsheet ##
    savePath = "G:\\Temp\\CocamideExtWithStartification_Fingerprints_test_predict_err.csv"
    CSV.write(savePath, inputCocamidesTest)


## predict Model 1 training set used in Model 2
    ## define functions
    function smilesid2RI(df, i)
        ID = df[i, "SMILES"]
        idx = findall(inputAllFPDB.SMILES .== ID)
        if (size(idx, 1) == 0)
            return float(0)
        end
        return inputAllFPDB[idx[end:end], "predictRi"][1]
    end
    #
    function smilesid2pubchemid(df, i)
        ID = df[i, "SMILES"]
        idx = findall(inputAllFPDB.SMILES .== ID)
        if (size(idx, 1) == 0)
            return ""
        end
        return inputAllFPDB[idx[end:end], "INCHIKEY"][1]
    end
    #
    ## import csv ##
    # 30684 x 793 df, columns include 
        #SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs, Predicted Expected RI
        inputAllFPDB = CSV.read("G:\\Temp\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
        sort!(inputAllFPDB, [:INCHIKEY, :SMILES])

        inputDB = CSV.read("G:\\Temp\\dataframe73_dfTrainSetWithStratification_withCNLPredictedRi.csv", DataFrame)
        sort!(inputDB, [:INCHIKEY])

        inputDB2 = CSV.read("G:\\Temp\\dataframe73_dfTestSetWithStratification_withCNLPredictedRi.csv", DataFrame)
        sort!(inputDB2, [:INCHIKEY])

        inputDB = vcat(inputDB, inputDB2)
        sort!(inputDB, [:INCHIKEY])

    # 5048 x 932 df 
    inputCocamidesTrain = CSV.read("G:\\Temp\\CocamideExtWithStartification_Fingerprints_train.csv", DataFrame)
    inputCocamidesTrain = inputCocamidesTrain[:, vcat(1,3)]
    sort!(inputCocamidesTrain, [:SMILES])
    inputCocamidesTrain[!, "dataset"] .= "train"
    inputCocamidesTrain[!, "MFpredictRI"] .= float(0)
    inputCocamidesTrain[!, "CNLpredictRI"] .= float(0)
    inputCocamidesTrain[!, "MFError"] .= float(0)
    inputCocamidesTrain[!, "MFAbsError"] .= float(0)
    inputCocamidesTrain[!, "CNLError"] .= float(0)
    inputCocamidesTrain[!, "CNLAbsError"] .= float(0)
    for i in 1:size(inputCocamidesTrain, 1)
        inputCocamidesTrain[i, "MFpredictRI"] = smilesid2RI(inputCocamidesTrain, i)
        inputCocamidesTrain[i, "MFError"] = inputCocamidesTrain[i, "MFpredictRI"] - inputCocamidesTrain[i, "RI"]
        inputCocamidesTrain[i, "MFAbsError"] = abs(inputCocamidesTrain[i, "MFError"])
        arr = findall(inputDB.INCHIKEY .== smilesid2pubchemid(inputCocamidesTrain, i))
        if size(arr, 1) > 0
            inputCocamidesTrain[i, "CNLpredictRI"] = inputDB[arr[end:end], "CNLpredictRi"][1]
            inputCocamidesTrain[i, "CNLError"] = inputCocamidesTrain[i, "CNLpredictRI"] - inputCocamidesTrain[i, "RI"]
            inputCocamidesTrain[i, "CNLAbsError"] = abs(inputCocamidesTrain[i, "CNLError"])
        end
    end
    inputCocamidesTrain = inputCocamidesTrain[inputCocamidesTrain.MFpredictRI .!= float(0), :]
    inputCocamidesTrain = inputCocamidesTrain[inputCocamidesTrain.CNLpredictRI .!= float(0), :]
    sort!(inputCocamidesTrain, [:MFAbsError])
    print(inputCocamidesTrain[1, "MFAbsError"])  # 0.12, 0.1175
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.25), "MFAbsError"])  # 32.58
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.5), "MFAbsError"])  # 73.62, 73.79
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.75), "MFAbsError"])  # 139.28
    print(inputCocamidesTrain[size(inputCocamidesTrain, 1)*1, "MFAbsError"])  # 723.20
    describe(inputCocamidesTrain)
    # avgAbsError = 103.06
    sort!(inputCocamidesTrain, [:MFError])
    print(inputCocamidesTrain[1, "MFError"])  # -723.20
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.25), "MFError"])  # -81.11
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.5), "MFError"])  # 12.55, 12.91
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.75), "MFError"])  # 71.66
    print(inputCocamidesTrain[size(inputCocamidesTrain, 1)*1, "MFError"])  # 390.58
    describe(inputCocamidesTrain)
    # avg = -9.83
    sort!(inputCocamidesTrain, [:CNLAbsError])
    print(inputCocamidesTrain[1, "CNLAbsError"])  # 0.31, 0.3073
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.25), "CNLAbsError"])  # 48.38
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.5), "CNLAbsError"])  # 108.45, 108.51
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.75), "CNLAbsError"])  # 192.19
    print(inputCocamidesTrain[size(inputCocamidesTrain, 1)*1, "CNLAbsError"])  # 818.03
    describe(inputCocamidesTrain)
    # avgAbsError = 137.65
    sort!(inputCocamidesTrain, [:CNLError])
    print(inputCocamidesTrain[1, "CNLError"])  # -818.03
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.25), "CNLError"])  # -85.83
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.5), "CNLError"])  # 31.44, 31.53
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.75), "CNLError"])  # 119.85
    print(inputCocamidesTrain[size(inputCocamidesTrain, 1)*1, "CNLError"])  # 768.51
    describe(inputCocamidesTrain)
    # avg = 12.67

    ## save the output table as a spreadsheet ##
    savePath = "G:\\Temp\\CocamideExtWithStartification_Fingerprints_train_predict_err2.csv"
    CSV.write(savePath, inputCocamidesTrain)


## predict Model 1 testing set used in Model 2
    # 1263 x 932 df 
    inputCocamidesTest = CSV.read("G:\\Temp\\CocamideExtWithStratification_Fingerprints_test.csv", DataFrame)
    inputCocamidesTest = inputCocamidesTest[:, vcat(1,3)]
    sort!(inputCocamidesTest, [:SMILES])
    inputCocamidesTest[!, "dataset"] .= "test"
    inputCocamidesTest[!, "MFpredictRI"] .= float(0)
    inputCocamidesTest[!, "CNLpredictRI"] .= float(0)
    inputCocamidesTest[!, "MFError"] .= float(0)
    inputCocamidesTest[!, "MFAbsError"] .= float(0)
    inputCocamidesTest[!, "CNLError"] .= float(0)
    inputCocamidesTest[!, "CNLAbsError"] .= float(0)
    for i in 1:size(inputCocamidesTest, 1)
        inputCocamidesTest[i, "MFpredictRI"] = smilesid2RI(inputCocamidesTest, i)
        inputCocamidesTest[i, "MFError"] = inputCocamidesTest[i, "MFpredictRI"] - inputCocamidesTest[i, "RI"]
        inputCocamidesTest[i, "MFAbsError"] = abs(inputCocamidesTest[i, "MFError"])
        arr = findall(inputDB.INCHIKEY .== smilesid2pubchemid(inputCocamidesTest, i))
        if size(arr, 1) > 0
            inputCocamidesTest[i, "CNLpredictRI"] = inputDB[arr[end:end], "CNLpredictRi"][1]
            inputCocamidesTest[i, "CNLError"] = inputCocamidesTest[i, "CNLpredictRI"] - inputCocamidesTest[i, "RI"]
            inputCocamidesTest[i, "CNLAbsError"] = abs(inputCocamidesTest[i, "CNLError"])
        end
    end
    inputCocamidesTest = inputCocamidesTest[inputCocamidesTest.MFpredictRI .!= float(0), :]
    inputCocamidesTest = inputCocamidesTest[inputCocamidesTest.CNLpredictRI .!= float(0), :]
    sort!(inputCocamidesTest, [:MFAbsError])
    print(inputCocamidesTest[1, "MFAbsError"])  # 1.80, 1.8023
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.25), "MFAbsError"])  # 29.27
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.5), "MFAbsError"])  # 82.23, 82.75
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.75), "MFAbsError"])  # 153.74
    print(inputCocamidesTest[size(inputCocamidesTest, 1)*1, "MFAbsError"])  # 741.76
    describe(inputCocamidesTest)
    # avgAbsError = 120.82
    sort!(inputCocamidesTest, [:MFError])
    print(inputCocamidesTest[1, "MFError"])  # -741.76
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.25), "MFError"])  # -99.72
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.5), "MFError"])  # 9.81, 10.31
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.75), "MFError"])  # 68.97
    print(inputCocamidesTest[size(inputCocamidesTest, 1)*1, "MFError"])  # 359.69
    describe(inputCocamidesTest)
    # avg = -20.83
    sort!(inputCocamidesTest, [:CNLAbsError])
    print(inputCocamidesTest[1, "CNLAbsError"])  # 1.79, 1.7852
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.25), "CNLAbsError"])  # 46.56
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.5), "CNLAbsError"])  # 100.80, 100.89
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.75), "CNLAbsError"])  # 203.44
    print(inputCocamidesTest[size(inputCocamidesTest, 1)*1, "CNLAbsError"])  # 668.82
    describe(inputCocamidesTest)
    # avgAbsError = 142.39
    sort!(inputCocamidesTest, [:CNLError])
    print(inputCocamidesTest[1, "CNLError"])  # -650.75
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.25), "CNLError"])  # -72.40
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.5), "CNLError"])  # 33.90, 34.00
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.75), "CNLError"])  # 112.81
    print(inputCocamidesTest[size(inputCocamidesTest, 1)*1, "CNLError"])  # 668.82
    describe(inputCocamidesTest)
    # avg = 9.00

    ## save the output table as a spreadsheet ##
    savePath = "G:\\Temp\\CocamideExtWithStartification_Fingerprints_test_predict_err2.csv"
    CSV.write(savePath, inputCocamidesTest)
    