VERSION
using Pkg
using CSV, DataFrames, Conda, LinearAlgebra, Statistics

# Train/Test Split by Leverage
using ProgressBars
using LinearAlgebra
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

# CNL model 95% leverage cut-off = 0.14604417882015916
# 180416 x 18 df
dfOutput1 = CSV.read("F:\\UvA\\app\\PestMix1-8_1ug-L_NoTea_1ul_AllIon_pos_39_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput4 = CSV.read("F:\\UvA\\app\\PestMix1-8_2-5ug-L_NoTea_1ul_AllIon_pos_51_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput7 = CSV.read("F:\\UvA\\app\\PestMix1-8_5ug-L_NoTea_1ul_AllIon_pos_19_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput10 = CSV.read("F:\\UvA\\app\\PestMix1-8_10ug-L_NoTea_1ul_AllIon_pos_37_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput13 = CSV.read("F:\\UvA\\app\\PestMix1-8_25ug-L_NoTea_1ul_AllIon_pos_15_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput16 = CSV.read("F:\\UvA\\app\\PestMix1-8_50ug-L_NoTea_1ul_AllIon_pos_31_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput19 = CSV.read("F:\\UvA\\app\\PestMix1-8_100ug-L_NoTea_1ul_AllIon_pos_28_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput22 = CSV.read("F:\\UvA\\app\\PestMix1-8_1000ug-L_NoTea_1ul_AllIon_pos_52_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput_noTea = vcat(dfOutput1, dfOutput4, dfOutput7, dfOutput10, dfOutput13, dfOutput16, dfOutput19, dfOutput22)
savePath = "F:\\UvA\\app\\allRealsampleNoTea_dataframeTPTNModeling.csv"
CSV.write(savePath, dfOutput_noTea)  # 18183 x 18

dfOutput2 = CSV.read("F:\\UvA\\app\\PestMix1-8_1ug-L_Tea_1-10dil_1ul_AllIon_pos_40_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput3 = CSV.read("F:\\UvA\\app\\PestMix1-8_1ug-L_Tea_1-100dil_1ul_AllIon_pos_20_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput5 = CSV.read("F:\\UvA\\app\\PestMix1-8_2-5ug-L_Tea_1-10dil_1ul_AllIon_pos_30_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput6 = CSV.read("F:\\UvA\\app\\PestMix1-8_2-5ug-L_Tea_1-100dil_1ul_AllIon_pos_17_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput8 = CSV.read("F:\\UvA\\app\\PestMix1-8_5ug-L_Tea_1-10dil_1ul_AllIon_pos_32_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput9 = CSV.read("F:\\UvA\\app\\PestMix1-8_5ug-L_Tea_1-100dil_1ul_AllIon_pos_53_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput11 = CSV.read("F:\\UvA\\app\\PestMix1-8_10ug-L_Tea_1-10dil_1ul_AllIon_pos_46_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput12 = CSV.read("F:\\UvA\\app\\PestMix1-8_10ug-L_Tea_1-100dil_1ul_AllIon_pos_34_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput14 = CSV.read("F:\\UvA\\app\\PestMix1-8_25ug-L_Tea_1-10dil_1ul_AllIon_pos_47_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput15 = CSV.read("F:\\UvA\\app\\PestMix1-8_25ug-L_Tea_1-100dil_1ul_AllIon_pos_45_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput17 = CSV.read("F:\\UvA\\app\\PestMix1-8_50ug-L_Tea_1-10dil_1ul_AllIon_pos_13_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput18 = CSV.read("F:\\UvA\\app\\PestMix1-8_50ug-L_Tea_1-100dil_1ul_AllIon_pos_12_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput20 = CSV.read("F:\\UvA\\app\\PestMix1-8_100ug-L_Tea_1-10dil_1ul_AllIon_pos_27_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput21 = CSV.read("F:\\UvA\\app\\PestMix1-8_100ug-L_Tea_1-100dil_1ul_AllIon_pos_8_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutput23 = CSV.read("F:\\UvA\\app\\PestMix1-8_1000ug-L_Tea_1-100dil_1ul_AllIon_pos_38_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
dfOutputPest = CSV.read("F:\\UvA\\PestMix1-8_1000ug-L_Tea_1-10dil_1ul_AllIon_pos_43_report_comp_IDs_dataframeTPTNModeling.csv", DataFrame)
describe(dfOutput23)
dfOutput_withTea = vcat(dfOutput2, dfOutput3, dfOutput5, dfOutput6, dfOutput8, dfOutput9, 
                    dfOutput11, dfOutput12, dfOutput14, dfOutput15, dfOutput17, dfOutput18, 
                    dfOutput20, dfOutput21, dfOutput23, dfOutputPest)
savePath = "F:\\UvA\\app\\allRealsampleWithTea_dataframeTPTNModeling.csv"
CSV.write(savePath, dfOutput_withTea)  # 129078 x 18

# training set, 3391333 x 21
inputDB = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainYesDFwithhl_all.csv", DataFrame)
sort!(inputDB, [:ENTRY])
describe(inputDB)
    # origin + 1-way
    inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
    inputDB = inputDB[inputDB.Leverage .<= 0.14604417882015916, :]
    for i = 1:size(inputDB, 1)
        inputDB[i, "MS1Error"] = abs(inputDB[i, "MS1Error"])
        inputDB[i, "DeltaRi"] = abs(inputDB[i, "DeltaRi"])
    end
    # save, ouputing 1686319 x 21 df, 0:1535009; 1:151310 = 
    savePath = "F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv"
    CSV.write(savePath, inputDB)
    inputDB[inputDB.LABEL .== 1, :]

    # DE
    insertcols!(inputDB, 10, ("MatchDiff"=>float(0)))
    inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
    inputDB = inputDB[inputDB.Leverage .<= 0.14604417882015916, :]
    for i = 1:size(inputDB, 1)
        inputDB[i, "RefMatchFragRatio"] = log10(inputDB[i, "RefMatchFragRatio"])
        inputDB[i, "UsrMatchFragRatio"] = log10(inputDB[i, "UsrMatchFragRatio"])
        inputDB[i, "FinalScoreRatio"] = log10(inputDB[i, "FinalScoreRatio"])
        inputDB[i, "MatchDiff"] = inputDB[i, "DirectMatch"] - inputDB[i, "ReversMatch"]
        inputDB[i, "MONOISOTOPICMASS"] = log10(inputDB[i, "MONOISOTOPICMASS"])
        if inputDB[i, "DeltaRi"] !== float(0)
            inputDB[i, "DeltaRi"] = inputDB[i, "DeltaRi"] * float(-1)
        end
    end
    # save, ouputing 1686319 x 21+1 df, 0:1535009; 1:151310 = 
    savePath = "F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv"
    CSV.write(savePath, inputDB)
    inputDB[inputDB.LABEL .== 1, :]

    # filter
    insertcols!(inputDB, 10, ("MatchDiff"=>float(0)))
    inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
    inputDB = inputDB[inputDB.Leverage .<= 0.14604417882015916, :]
    describe(inputDB[inputDB.LABEL .== 0, :])  #  0.000160256  -0.03                            0.0           0.03
    describe(inputDB[inputDB.LABEL .== 1, :])  # -3.30447e-8  -0.001                           0.0         0.001
    inputDB = inputDB[inputDB.MS1Error .>= float(-0.001), :]
    inputDB = inputDB[inputDB.MS1Error .<= float(0.001), :]
    for i = 1:size(inputDB, 1)
        inputDB[i, "RefMatchFragRatio"] = log10(inputDB[i, "RefMatchFragRatio"])
        inputDB[i, "UsrMatchFragRatio"] = log10(inputDB[i, "UsrMatchFragRatio"])
        inputDB[i, "FinalScoreRatio"] = log10(inputDB[i, "FinalScoreRatio"])
        inputDB[i, "MatchDiff"] = inputDB[i, "DirectMatch"] - inputDB[i, "ReversMatch"]
        inputDB[i, "MONOISOTOPICMASS"] = log10(inputDB[i, "MONOISOTOPICMASS"])
        if inputDB[i, "DeltaRi"] !== float(0)
            inputDB[i, "DeltaRi"] = inputDB[i, "DeltaRi"] * float(-1)
        end
    end
    # save, ouputing 485631 x 21+1 df, 0:334321; 1:151310 = 
    savePath = "F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilter.csv"
    CSV.write(savePath, inputDB)
    inputDB[inputDB.LABEL .== 1, :]

    # std
    insertcols!(inputDB, 10, ("MatchDiff"=>float(0)))
    inputDB = inputDB[inputDB.FinalScoreRatio .>= float(0.5), :]
    inputDB = inputDB[inputDB.Leverage .<= 0.14604417882015916, :]
    describe(inputDB[inputDB.LABEL .== 0, :])
    describe(inputDB[inputDB.LABEL .== 1, :])
    inputDB = inputDB[inputDB.MS1Error .>= float(-0.001), :]
    inputDB = inputDB[inputDB.MS1Error .<= float(0.001), :]
    for i = 1:size(inputDB, 1)
        inputDB[i, "RefMatchFragRatio"] = log10(inputDB[i, "RefMatchFragRatio"])
        inputDB[i, "UsrMatchFragRatio"] = log10(inputDB[i, "UsrMatchFragRatio"])
        inputDB[i, "FinalScoreRatio"] = log10(inputDB[i, "FinalScoreRatio"])
        inputDB[i, "MatchDiff"] = inputDB[i, "DirectMatch"] - inputDB[i, "ReversMatch"]
        inputDB[i, "MONOISOTOPICMASS"] = log10(inputDB[i, "MONOISOTOPICMASS"])
        if inputDB[i, "DeltaRi"] !== float(0)
            inputDB[i, "DeltaRi"] = inputDB[i, "DeltaRi"] * float(-1)
        end
    end
    describe(inputDB[:, 5:14])
    for f = 5:14
        avg = float(mean(inputDB[:, f]))
        top = float(maximum(inputDB[:, f]))
        down = float(minimum(inputDB[:, f]))
        for i = 1:size(inputDB, 1)
            inputDB[i, f] = (inputDB[i, f] - avg) / (top - down)
        end
    end
    # save, ouputing 485631 x 21+1 df, 0:334321; 1:151310 = 
    savePath = "F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilterSTD.csv"
    CSV.write(savePath, inputDB)
    inputDB[inputDB.LABEL .== 1, :]


# testing set, 847838 x 21
inputDB_test = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestYesDFwithhl_all.csv", DataFrame)
sort!(inputDB_test, [:ENTRY])
describe(inputDB_test)
    # origin + 1-way
    inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
    inputDB_test = inputDB_test[inputDB_test.Leverage .<= 0.14604417882015916, :]
    for i = 1:size(inputDB_test, 1)
        inputDB_test[i, "MS1Error"] = abs(inputDB_test[i, "MS1Error"])
        inputDB_test[i, "DeltaRi"] = abs(inputDB_test[i, "DeltaRi"])
    end
    # save, ouputing 421381 x 21 df, 0:383416; 1:37965 = 
    savePath = "F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv"
    CSV.write(savePath, inputDB_test)
    inputDB_test[inputDB_test.LABEL .== 1, :]

    # DE
    insertcols!(inputDB_test, 10, ("MatchDiff"=>float(0)))
    inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
    inputDB_test = inputDB_test[inputDB_test.Leverage .<= 0.14604417882015916, :]
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
    # save, ouputing 421381 x 21+1 df, 0:383416; 1:37965 = 
    savePath = "F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv"
    CSV.write(savePath, inputDB_test)
    inputDB_test[inputDB_test.LABEL .== 1, :]

    # filter
    insertcols!(inputDB_test, 10, ("MatchDiff"=>float(0)))
    inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
    inputDB_test = inputDB_test[inputDB_test.Leverage .<= 0.14604417882015916, :]
    describe(inputDB_test[inputDB_test.LABEL .== 0, :])  #   0.000167617  -0.03                            0.0          0.03
    describe(inputDB_test[inputDB_test.LABEL .== 1, :])  # 5.79481e-7  -0.001                           0.0         0.001
    inputDB_test = inputDB_test[inputDB_test.MS1Error .>= float(-0.001), :]
    inputDB_test = inputDB_test[inputDB_test.MS1Error .<= float(0.001), :]
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
    # save, ouputing 121946 x 21+1 df, 0:83981; 1:37965 = 
    savePath = "F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilter.csv"
    CSV.write(savePath, inputDB_test)
    inputDB_test[inputDB_test.LABEL .== 1, :]

    # std
    insertcols!(inputDB_test, 10, ("MatchDiff"=>float(0)))
    inputDB_test = inputDB_test[inputDB_test.FinalScoreRatio .>= float(0.5), :]
    inputDB_test = inputDB_test[inputDB_test.Leverage .<= 0.14604417882015916, :]
    describe(inputDB_test[inputDB_test.LABEL .== 0, :])
    describe(inputDB_test[inputDB_test.LABEL .== 1, :])
    inputDB_test = inputDB_test[inputDB_test.MS1Error .>= float(-0.001), :]
    inputDB_test = inputDB_test[inputDB_test.MS1Error .<= float(0.001), :]
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
    # save, ouputing 121946 x 21+1 df, 0:83981; 1:37965 = 
    savePath = "F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilterSTD.csv"
    CSV.write(savePath, inputDB_test)
    inputDB_test[inputDB_test.LABEL .== 1, :]

# real sample set (no tea)
dfOutput_noTea = CSV.read("F:\\UvA\\app\\allRealsampleNoTea_dataframeTPTNModeling.csv", DataFrame)
sort!(dfOutput_noTea, [:ENTRY])
describe(dfOutput_noTea)
    # origin + 1-way
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.FinalScoreRatio .>= float(0.5), :]
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.Leverage .<= 0.14604417882015916, :]
    for i = 1:size(dfOutput_noTea, 1)
        dfOutput_noTea[i, "MS1Error"] = abs(dfOutput_noTea[i, "MS1Error"])
        dfOutput_noTea[i, "DeltaRi"] = abs(dfOutput_noTea[i, "DeltaRi"])
    end
    # save, ouputing 10908 x 18 df, 0:7173; 1:3735 = 
    savePath = "F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv"
    CSV.write(savePath, dfOutput_noTea)
    dfOutput_noTea[dfOutput_noTea.LABEL .== 1, :]

    # DE
    insertcols!(dfOutput_noTea, 10, ("MatchDiff"=>float(0)))
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.FinalScoreRatio .>= float(0.5), :]
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.Leverage .<= 0.14604417882015916, :]
    for i = 1:size(dfOutput_noTea, 1)
        dfOutput_noTea[i, "RefMatchFragRatio"] = log10(dfOutput_noTea[i, "RefMatchFragRatio"])
        dfOutput_noTea[i, "UsrMatchFragRatio"] = log10(dfOutput_noTea[i, "UsrMatchFragRatio"])
        dfOutput_noTea[i, "FinalScoreRatio"] = log10(dfOutput_noTea[i, "FinalScoreRatio"])
        dfOutput_noTea[i, "MatchDiff"] = dfOutput_noTea[i, "DirectMatch"] - dfOutput_noTea[i, "ReversMatch"]
        dfOutput_noTea[i, "MONOISOTOPICMASS"] = log10(dfOutput_noTea[i, "MONOISOTOPICMASS"])
        if dfOutput_noTea[i, "DeltaRi"] !== float(0)
            dfOutput_noTea[i, "DeltaRi"] = dfOutput_noTea[i, "DeltaRi"] * float(-1)
        end
    end
    # save, ouputing 10908 x 18+1 df, 0:7173; 1:3735 = 
    savePath = "F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv"
    CSV.write(savePath, dfOutput_noTea)
    dfOutput_noTea[dfOutput_noTea.LABEL .== 1, :]

    # filter
    insertcols!(dfOutput_noTea, 10, ("MatchDiff"=>float(0)))
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.FinalScoreRatio .>= float(0.5), :]
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.Leverage .<= 0.14604417882015916, :]
    describe(dfOutput_noTea[dfOutput_noTea.LABEL .== 0, :])  #  0.000332218  -0.067                             -0.004     0.064
    describe(dfOutput_noTea[dfOutput_noTea.LABEL .== 1, :])  # -0.00625221  -0.046                             -0.004      0.066
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.MS1Error .>= float(-0.046), :]
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.MS1Error .<= float(0.066), :]
    for i = 1:size(dfOutput_noTea, 1)
        dfOutput_noTea[i, "RefMatchFragRatio"] = log10(dfOutput_noTea[i, "RefMatchFragRatio"])
        dfOutput_noTea[i, "UsrMatchFragRatio"] = log10(dfOutput_noTea[i, "UsrMatchFragRatio"])
        dfOutput_noTea[i, "FinalScoreRatio"] = log10(dfOutput_noTea[i, "FinalScoreRatio"])
        dfOutput_noTea[i, "MatchDiff"] = dfOutput_noTea[i, "DirectMatch"] - dfOutput_noTea[i, "ReversMatch"]
        dfOutput_noTea[i, "MONOISOTOPICMASS"] = log10(dfOutput_noTea[i, "MONOISOTOPICMASS"])
        if dfOutput_noTea[i, "DeltaRi"] !== float(0)
            dfOutput_noTea[i, "DeltaRi"] = dfOutput_noTea[i, "DeltaRi"] * float(-1)
        end
    end
    # save, ouputing 10868 x 18+1 df, 0:7133; 1:3735 = 
    savePath = "F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilter.csv"
    CSV.write(savePath, dfOutput_noTea)
    dfOutput_noTea[dfOutput_noTea.LABEL .== 1, :]

    # std
    insertcols!(dfOutput_noTea, 10, ("MatchDiff"=>float(0)))
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.FinalScoreRatio .>= float(0.5), :]
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.Leverage .<= 0.14604417882015916, :]
    describe(dfOutput_noTea[dfOutput_noTea.LABEL .== 0, :])
    describe(dfOutput_noTea[dfOutput_noTea.LABEL .== 1, :])
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.MS1Error .>= float(-0.046), :]
    dfOutput_noTea = dfOutput_noTea[dfOutput_noTea.MS1Error .<= float(0.066), :]
    for i = 1:size(dfOutput_noTea, 1)
        dfOutput_noTea[i, "RefMatchFragRatio"] = log10(dfOutput_noTea[i, "RefMatchFragRatio"])
        dfOutput_noTea[i, "UsrMatchFragRatio"] = log10(dfOutput_noTea[i, "UsrMatchFragRatio"])
        dfOutput_noTea[i, "FinalScoreRatio"] = log10(dfOutput_noTea[i, "FinalScoreRatio"])
        dfOutput_noTea[i, "MatchDiff"] = dfOutput_noTea[i, "DirectMatch"] - dfOutput_noTea[i, "ReversMatch"]
        dfOutput_noTea[i, "MONOISOTOPICMASS"] = log10(dfOutput_noTea[i, "MONOISOTOPICMASS"])
        if dfOutput_noTea[i, "DeltaRi"] !== float(0)
            dfOutput_noTea[i, "DeltaRi"] = dfOutput_noTea[i, "DeltaRi"] * float(-1)
        end
    end
    describe(dfOutput_noTea[:, 5:14])
    for f = 5:14
        avg = float(mean(dfOutput_noTea[:, f]))
        top = float(maximum(dfOutput_noTea[:, f]))
        down = float(minimum(dfOutput_noTea[:, f]))
        for i = 1:size(dfOutput_noTea, 1)
            dfOutput_noTea[i, f] = (dfOutput_noTea[i, f] - avg) / (top - down)
        end
    end
    # save, ouputing 10868 x 18+1 df, 0:7133; 1:3735 = 
    savePath = "F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilterSTD.csv"
    CSV.write(savePath, dfOutput_noTea)
    dfOutput_noTea[dfOutput_noTea.LABEL .== 1, :]

# real sample set (with tea)
dfOutput_Tea = CSV.read("F:\\UvA\\app\\allRealsampleWithTea_dataframeTPTNModeling.csv", DataFrame)
sort!(dfOutput_Tea, [:ENTRY])
describe(dfOutput_Tea)
    # origin + 1-way
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.FinalScoreRatio .>= float(0.5), :]
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.Leverage .<= 0.14604417882015916, :]
    for i = 1:size(dfOutput_Tea, 1)
        dfOutput_Tea[i, "MS1Error"] = abs(dfOutput_Tea[i, "MS1Error"])
        dfOutput_Tea[i, "DeltaRi"] = abs(dfOutput_Tea[i, "DeltaRi"])
    end
    # save, ouputing 29599 x 18 df, 0:21412; 1:8187 = 
    savePath = "F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv"
    CSV.write(savePath, dfOutput_Tea)
    dfOutput_Tea[dfOutput_Tea.LABEL .== 1, :]

    # DE
    insertcols!(dfOutput_Tea, 10, ("MatchDiff"=>float(0)))
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.FinalScoreRatio .>= float(0.5), :]
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.Leverage .<= 0.14604417882015916, :]
    for i = 1:size(dfOutput_Tea, 1)
        dfOutput_Tea[i, "RefMatchFragRatio"] = log10(dfOutput_Tea[i, "RefMatchFragRatio"])
        dfOutput_Tea[i, "UsrMatchFragRatio"] = log10(dfOutput_Tea[i, "UsrMatchFragRatio"])
        dfOutput_Tea[i, "FinalScoreRatio"] = log10(dfOutput_Tea[i, "FinalScoreRatio"])
        dfOutput_Tea[i, "MatchDiff"] = dfOutput_Tea[i, "DirectMatch"] - dfOutput_Tea[i, "ReversMatch"]
        dfOutput_Tea[i, "MONOISOTOPICMASS"] = log10(dfOutput_Tea[i, "MONOISOTOPICMASS"])
        if dfOutput_Tea[i, "DeltaRi"] !== float(0)
            dfOutput_Tea[i, "DeltaRi"] = dfOutput_Tea[i, "DeltaRi"] * float(-1)
        end
    end
    # save, ouputing 29599 x 18+1 df, 0:21412; 1:8187 = 
    savePath = "F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv"
    CSV.write(savePath, dfOutput_Tea)
    dfOutput_Tea[dfOutput_Tea.LABEL .== 1, :]

    # filter
    insertcols!(dfOutput_Tea, 10, ("MatchDiff"=>float(0)))
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.FinalScoreRatio .>= float(0.5), :]
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.Leverage .<= 0.14604417882015916, :]
    describe(dfOutput_Tea[dfOutput_Tea.LABEL .== 0, :])  #  0.00411905   -0.068                             0.0        0.068
    describe(dfOutput_Tea[dfOutput_Tea.LABEL .== 1, :])  # -0.00604593  -0.061                             -0.004     0.058
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.MS1Error .>= float(-0.061), :]
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.MS1Error .<= float(0.058), :]
    for i = 1:size(dfOutput_Tea, 1)
        dfOutput_Tea[i, "RefMatchFragRatio"] = log10(dfOutput_Tea[i, "RefMatchFragRatio"])
        dfOutput_Tea[i, "UsrMatchFragRatio"] = log10(dfOutput_Tea[i, "UsrMatchFragRatio"])
        dfOutput_Tea[i, "FinalScoreRatio"] = log10(dfOutput_Tea[i, "FinalScoreRatio"])
        dfOutput_Tea[i, "MatchDiff"] = dfOutput_Tea[i, "DirectMatch"] - dfOutput_Tea[i, "ReversMatch"]
        dfOutput_Tea[i, "MONOISOTOPICMASS"] = log10(dfOutput_Tea[i, "MONOISOTOPICMASS"])
        if dfOutput_Tea[i, "DeltaRi"] !== float(0)
            dfOutput_Tea[i, "DeltaRi"] = dfOutput_Tea[i, "DeltaRi"] * float(-1)
        end
    end
    # save, ouputing 29397 x 18+1 df, 0:21210; 1:8187 = 
    savePath = "F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilter.csv"
    CSV.write(savePath, dfOutput_Tea)
    dfOutput_Tea[dfOutput_Tea.LABEL .== 1, :]

    # std
    insertcols!(dfOutput_Tea, 10, ("MatchDiff"=>float(0)))
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.FinalScoreRatio .>= float(0.5), :]
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.Leverage .<= 0.14604417882015916, :]
    describe(dfOutput_Tea[dfOutput_Tea.LABEL .== 0, :])
    describe(dfOutput_Tea[dfOutput_Tea.LABEL .== 1, :])
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.MS1Error .>= float(-0.061), :]
    dfOutput_Tea = dfOutput_Tea[dfOutput_Tea.MS1Error .<= float(0.058), :]
    for i = 1:size(dfOutput_Tea, 1)
        dfOutput_Tea[i, "RefMatchFragRatio"] = log10(dfOutput_Tea[i, "RefMatchFragRatio"])
        dfOutput_Tea[i, "UsrMatchFragRatio"] = log10(dfOutput_Tea[i, "UsrMatchFragRatio"])
        dfOutput_Tea[i, "FinalScoreRatio"] = log10(dfOutput_Tea[i, "FinalScoreRatio"])
        dfOutput_Tea[i, "MatchDiff"] = dfOutput_Tea[i, "DirectMatch"] - dfOutput_Tea[i, "ReversMatch"]
        dfOutput_Tea[i, "MONOISOTOPICMASS"] = log10(dfOutput_Tea[i, "MONOISOTOPICMASS"])
        if dfOutput_Tea[i, "DeltaRi"] !== float(0)
            dfOutput_Tea[i, "DeltaRi"] = dfOutput_Tea[i, "DeltaRi"] * float(-1)
        end
    end
    describe(dfOutput_Tea[:, 5:14])
    for f = 5:14
        avg = float(mean(dfOutput_Tea[:, f]))
        top = float(maximum(dfOutput_Tea[:, f]))
        down = float(minimum(dfOutput_Tea[:, f]))
        for i = 1:size(dfOutput_Tea, 1)
            dfOutput_Tea[i, f] = (dfOutput_Tea[i, f] - avg) / (top - down)
        end
    end
    # save, ouputing 29397 x 18+1 df, 0:21210; 1:8187 = 
    savePath = "F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilterSTD.csv"
    CSV.write(savePath, dfOutput_Tea)
    dfOutput_Tea[dfOutput_Tea.LABEL .== 1, :]
