## INPUT(S)
# trainDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
# trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
# trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
# testDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
# testDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
# testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
# noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
# noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
# noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
# TeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
# TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
# TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***

## OUTPUT(S)
# outplot_TPTNDistrution_FeatureMonoisotopicMass_noFilter.png

VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using ScikitLearn
using StatsPlots
using Plots

## import training set ##
# 1686319/1686319 / 485631/485631 x 21 / 22 / 22 / 22
trainDf = CSV.read("F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv", DataFrame)
trainDEDf = CSV.read("F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv", DataFrame)
#trainDEFDf = CSV.read("F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilter.csv", DataFrame)
trainDEFSDf = CSV.read("F:\\UvA\\app\\trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)

## import testing set ##
# 421381/421381 / 121946/121946 x 21 / 22 / 22 / 22
testDf = CSV.read("F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv", DataFrame)
testDEDf = CSV.read("F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv", DataFrame)
#testDEFDf = CSV.read("F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilter.csv", DataFrame)
testDEFSDf = CSV.read("F:\\UvA\\app\\testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)

## import validation set ##, spike blank (No Tea)
# 10908/10908 / 10868/10868 x 18 / 19 / 19 / 19
noTeaDf = CSV.read("F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv", DataFrame)
noTeaDEDf = CSV.read("F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv", DataFrame)
#noTeaDEFDf = CSV.read("F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilter.csv", DataFrame)
noTeaDEFSDf = CSV.read("F:\\UvA\\app\\noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)

## import real sample set ## (With Tea)
# 29599/29599 / 29397/29397 x 18 / 19 / 19 / 19
TeaDf = CSV.read("F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv", DataFrame)
TeaDEDf = CSV.read("F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv", DataFrame)
#TeaDEFDf = CSV.read("F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEFilter.csv", DataFrame)
TeaDEFSDf = CSV.read("F:\\UvA\\app\\TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv", DataFrame)


# ==================================================================================================
## assign variables for TP and TN data ##
trainDf_0 = trainDf[trainDf.LABEL .== 0, :]
trainDf_1 = trainDf[trainDf.LABEL .== 1, :]
testDf_0 = testDf[testDf.LABEL .== 0, :]
testDf_1 = testDf[testDf.LABEL .== 1, :]
noTeaDf_0 = noTeaDf[noTeaDf.LABEL .== 0, :]
noTeaDf_1 = noTeaDf[noTeaDf.LABEL .== 1, :]
TeaDf_1 = TeaDf[TeaDf.LABEL .== 1, :]

trainDEDf_0 = trainDEDf[trainDEDf.LABEL .== 0, :]
trainDEDf_1 = trainDEDf[trainDEDf.LABEL .== 1, :]
testDEDf_0 = testDEDf[testDEDf.LABEL .== 0, :]
testDEDf_1 = testDEDf[testDEDf.LABEL .== 1, :]
noTeaDEDf_0 = noTeaDEDf[noTeaDEDf.LABEL .== 0, :]
noTeaDEDf_1 = noTeaDEDf[noTeaDEDf.LABEL .== 1, :]
TeaDEDf_1 = TeaDEDf[TeaDEDf.LABEL .== 1, :]

#= trainDEFDf_0 = trainDEFDf[trainDEFDf.LABEL .== 0, :]
trainDEFDf_1 = trainDEFDf[trainDEFDf.LABEL .== 1, :]
testDEFDf_0 = testDEFDf[testDEFDf.LABEL .== 0, :]
testDEFDf_1 = testDEFDf[testDEFDf.LABEL .== 1, :]
noTeaDEFDf_0 = noTeaDEFDf[noTeaDEFDf.LABEL .== 0, :]
noTeaDEFDf_1 = noTeaDEFDf[noTeaDEFDf.LABEL .== 1, :]
TeaDEFDf_1 = TeaDEFDf[TeaDEFDf.LABEL .== 1, :] =#

trainDEFSDf_0 = trainDEFSDf[trainDEFSDf.LABEL .== 0, :]
trainDEFSDf_1 = trainDEFSDf[trainDEFSDf.LABEL .== 1, :]
testDEFSDf_0 = testDEFSDf[testDEFSDf.LABEL .== 0, :]
testDEFSDf_1 = testDEFSDf[testDEFSDf.LABEL .== 1, :]
noTeaDEFSDf_0 = noTeaDEFSDf[noTeaDEFSDf.LABEL .== 0, :]
noTeaDEFSDf_1 = noTeaDEFSDf[noTeaDEFSDf.LABEL .== 1, :]
TeaDEFSDf_1 = TeaDEFSDf[TeaDEFSDf.LABEL .== 1, :]


# ==================================================================================================
## plot graph ##
using DataSci4Chem
#
layout = @layout [a{0.33w,0.25h} b{0.33w,0.25h} c{0.33w,0.25h} 
                  d{0.33w,0.25h} e{0.33w,0.25h} f{0.33w,0.25h} 
                  g{0.33w,0.25h} h{0.33w,0.25h} i{0.33w,0.25h} 
                  j{0.33w,0.25h} k{0.33w,0.25h} l{0.33w,0.25h}]
default(grid = false, legend = false)
gr()
#
outplotTPTNdetaRiDistrution = plot(layout = layout, link = :both, 
        size = (1500, 1500), margin = (8, :mm), dpi = 300)
#
histogram!(trainDf_0[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MonoisotopicMass/1000", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(trainDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MonoisotopicMass/1000", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDf_0[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MonoisotopicMass/1000", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(testDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MonoisotopicMass/1000", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(noTeaDf_0[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 7, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MonoisotopicMass/1000", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Validation Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(noTeaDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 7, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MonoisotopicMass/1000", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Validation Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(TeaDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 10, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MonoisotopicMass/1000", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Real Sample Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(trainDEDf_0[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(trainDEDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDEDf_0[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nTesting Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(testDEDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nTesting Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(noTeaDEDf_0[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 8, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nValidation Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(noTeaDEDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 8, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nValidation Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(TeaDEDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 11, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nReal Sample Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(trainDEFSDf_0[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of\nlog(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(trainDEFSDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of\nlog(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDEFSDf_0[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of\nlog(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nTesting Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(testDEFSDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of\nlog(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nTesting Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(noTeaDEFSDf_0[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 9, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of\nlog(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nValidation Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(noTeaDEFSDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 9, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of\nlog(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nValidation Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(TeaDEFSDf_1[:, "MONOISOTOPICMASS"], bins = 150, 
    subplot = 12, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of\nlog(MonoisotopicMass/1000)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nReal Sample Dataset", 
    titlefont = font(12), 
    dpi = 300)

## save ##
savefig(outplotTPTNdetaRiDistrution, "F:\\UvA\\app\\outplot_TPTNDistrution_FeatureMonoisotopicMass_noFilter.png")
