VERSION
using Pkg
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using ScikitLearn
using StatsPlots
using Plots

# 4103848 x 4 + 8 + 1 + 2 + 1 + 1
trainDf = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl0d5FinalScoreRatio2.csv", DataFrame)
trainDEDf = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl0d5FinalScoreRatio2DE.csv", DataFrame)
trainDEDf2 = CSV.read("F:\\UvA\\dataframeTPTNModeling_TrainDFwithhl0d5FinalScoreRatio2DE2.csv", DataFrame)

testDf = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDFwithhl0d5FinalScoreRatio2.csv", DataFrame)
testDEDf = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDFwithhl0d5FinalScoreRatio2DE.csv", DataFrame)
testDEDf2 = CSV.read("F:\\UvA\\dataframeTPTNModeling_TestDFwithhl0d5FinalScoreRatio2DE2.csv", DataFrame)

pestDf = CSV.read("F:\\UvA\\dataframeTPTNModeling_pestDFwithhl0d5FinalScoreRatio2.csv", DataFrame)
pestDEDf = CSV.read("F:\\UvA\\dataframeTPTNModeling_pestDFwithhl0d5FinalScore2RatioDE.csv", DataFrame)
pestDEDf2 = CSV.read("F:\\UvA\\dataframeTPTNModeling_pestDFwithhl0d5FinalScore2RatioDE2.csv", DataFrame)

describe(trainDf)[1:end, :]
describe(testDf)[1:end, :]
describe(pestDf_DE0)[1:end, :]
# ==================================================================================================

trainDf_0 = trainDf[trainDf.LABEL .== 0, :]
trainDf_1 = trainDf[trainDf.LABEL .== 1, :]
testDf_0 = testDf[testDf.LABEL .== 0, :]
testDf_1 = testDf[testDf.LABEL .== 1, :]
pestDf_0 = pestDf[pestDf.LABEL .== 0, :]
pestDf_1 = pestDf[pestDf.LABEL .== 1, :]

trainDf_DE0 = trainDEDf[trainDEDf.LABEL .== 0, :]
trainDf_DE1 = trainDEDf[trainDEDf.LABEL .== 1, :]
testDf_DE0 = testDEDf[testDEDf.LABEL .== 0, :]
testDf_DE1 = testDEDf[testDEDf.LABEL .== 1, :]
pestDf_DE0 = pestDEDf[pestDEDf.LABEL .== 0, :]
pestDf_DE1 = pestDEDf[pestDEDf.LABEL .== 1, :]

trainDf2_DE0 = trainDEDf2[trainDEDf2.LABEL .== 0, :]
trainDf2_DE1 = trainDEDf2[trainDEDf2.LABEL .== 1, :]
testDf2_DE0 = testDEDf2[testDEDf2.LABEL .== 0, :]
testDf2_DE1 = testDEDf2[testDEDf2.LABEL .== 1, :]
pestDf2_DE0 = pestDEDf2[pestDEDf2.LABEL .== 0, :]
pestDf2_DE1 = pestDEDf2[pestDEDf2.LABEL .== 1, :]
# ==================================================================================================

using DataSci4Chem

layout = @layout [a{0.33w,0.33h} b{0.33w,0.33h} c{0.33w,0.33h}
                  d{0.33w,0.33h} e{0.33w,0.33h} f{0.33w,0.33h}
                  g{0.33w,0.33h} h{0.33w,0.33h} i{0.33w,0.33h}]
default(grid = false, legend = false)
gr()

outplotTPTNdetaRiDistrution = plot(layout = layout, link = :both, 
        size = (1800, 1200), margin = (8, :mm), dpi = 300)

histogram!(trainDf_0[:, "UsrMatchFragRatio"], 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "UsrMatchFragRatio", xguidefontsize=10, 
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
histogram!(trainDf_1[:, "UsrMatchFragRatio"], 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "UsrMatchFragRatio", xguidefontsize=10, 
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

histogram!(trainDf_DE0[:, "UsrMatchFragRatio"], 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(trainDf_DE1[:, "UsrMatchFragRatio"], 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Training Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(trainDf2_DE0[:, "UsrMatchFragRatio"], 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Filtered Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(trainDf2_DE1[:, "UsrMatchFragRatio"], 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Filtered Training Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(testDf_0[:, "UsrMatchFragRatio"], 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "UsrMatchFragRatio", xguidefontsize=10, 
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
histogram!(testDf_1[:, "UsrMatchFragRatio"], 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "UsrMatchFragRatio", xguidefontsize=10, 
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

histogram!(testDf_DE0[:, "UsrMatchFragRatio"], 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDf_DE1[:, "UsrMatchFragRatio"], 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(testDf2_DE0[:, "UsrMatchFragRatio"], 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Filtered Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDf2_DE1[:, "UsrMatchFragRatio"], 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Filtered Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(pestDf_0[:, "UsrMatchFragRatio"], 
    subplot = 7, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "UsrMatchFragRatio", xguidefontsize=10, 
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
    title = "External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf_1[:, "UsrMatchFragRatio"], 
    subplot = 7, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "UsrMatchFragRatio", xguidefontsize=10, 
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
    title = "External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(pestDf_DE0[:, "UsrMatchFragRatio"], 
    subplot = 8, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf_DE1[:, "UsrMatchFragRatio"], 
    subplot = 8, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(pestDf2_DE0[:, "UsrMatchFragRatio"], 
    subplot = 9, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Filtered External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf2_DE1[:, "UsrMatchFragRatio"], 
    subplot = 9, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "log(UsrMatchFragRatio)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Filtered External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

# Saving
savefig(outplotTPTNdetaRiDistrution, "F:\\UvA\\outplot_TPTNDistrution_UsrMatchFragRatio2.png")

# ==================================================================================================

layout = @layout [a{0.33w,0.33h} b{0.33w,0.33h} c{0.33w,0.33h}
                  d{0.33w,0.33h} e{0.33w,0.33h} f{0.33w,0.33h}
                  g{0.33w,0.33h} h{0.33w,0.33h} i{0.33w,0.33h}]
default(grid = false, legend = false)
gr()

outplotTPTNdetaRiDistrution = plot(layout = layout, link = :both, 
        size = (1800, 1200), margin = (8, :mm), dpi = 300)

histogram!(trainDf_0[:, "MS1Error"], 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "1-Way MS1Error", xguidefontsize=10, 
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
    title = "Preprocessed Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(trainDf_1[:, "MS1Error"], 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "1-Way MS1Error", xguidefontsize=10, 
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
    title = "Preprocessed Training Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(trainDf_DE0[:, "MS1Error"], 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "2-Way MS1Error", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(trainDf_DE1[:, "MS1Error"], 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "2-Way MS1Error", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(trainDf2_DE0[:, "MS1Error"], 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "2-Way MS1Error", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Filtered Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(trainDf2_DE1[:, "MS1Error"], 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "2-Way MS1Error", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Filtered Training Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(testDf_0[:, "MS1Error"], 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "1-Way MS1Error", xguidefontsize=10, 
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
    title = "Preprocessed Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDf_1[:, "MS1Error"], 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "1-Way MS1Error", xguidefontsize=10, 
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
    title = "Preprocessed Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(testDf_DE0[:, "MS1Error"], 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "2-Way MS1Error", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDf_DE1[:, "DeltaRi"], 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "2-Way DeltaRI", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(pestDf_0[:, "DeltaRi"], 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "1-Way DeltaRI", xguidefontsize=10, 
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
    title = "Preprocessed External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf_1[:, "DeltaRi"], 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "1-Way DeltaRI", xguidefontsize=10, 
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
    title = "Preprocessed External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(pestDf_DE0[:, "DeltaRi"], 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "2-Way DeltaRI", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf_DE1[:, "DeltaRi"], 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "2-Way DeltaRI", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

# Saving
savefig(outplotTPTNdetaRiDistrution, "F:\\UvA\\outplot_TPTNDistrution_DeltaRI.png")

# ==================================================================================================

layout = @layout [a{0.50w,0.33h} b{0.50w,0.33h}
                  c{0.50w,0.33h} d{0.50w,0.33h}
                  e{0.50w,0.33h} f{0.50w,0.33h}]
default(grid = false, legend = false)
gr()

outplotTPTNdetaRiDistrution = plot(layout = layout, link = :both, 
        size = (1200, 1200), margin = (8, :mm), dpi = 300)

histogram!(trainDf_0[:, "MS2Error"], 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2Error", xguidefontsize=10, 
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
histogram!(trainDf_1[:, "MS2Error"], 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2Error", xguidefontsize=10, 
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

histogram!(trainDf_DE0[:, "MS2ErrorStd"], 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2ErrorStd", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(trainDf_DE1[:, "MS2ErrorStd"], 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2ErrorStd", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(testDf_0[:, "MS2Error"], 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2Error", xguidefontsize=10, 
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
histogram!(testDf_1[:, "MS2Error"], 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2Error", xguidefontsize=10, 
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

histogram!(testDf_DE0[:, "MS2ErrorStd"], 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2ErrorStd", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDf_DE1[:, "MS2ErrorStd"], 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2ErrorStd", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(pestDf_0[:, "MS2Error"], 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2Error", xguidefontsize=10, 
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
    title = "External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf_1[:, "MS2Error"], 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2Error", xguidefontsize=10, 
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
    title = "External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(pestDf_DE0[:, "MS2ErrorStd"], 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2ErrorStd", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf_DE1[:, "MS2ErrorStd"], 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MS2ErrorStd", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "External Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

# Saving
savefig(outplotTPTNdetaRiDistrution, "F:\\UvA\\outplot_TPTNDistrution_MS2ErrorNStd.png")

# ==================================================================================================

layout = @layout [a{0.33w,0.33h} b{0.33w,0.33h} c{0.33w,0.33h}
                  d{0.33w,0.33h} e{0.33w,0.33h} f{0.33w,0.33h}
                  g{0.33w,0.33h} h{0.33w,0.33h} i{0.33w,0.33h}]
default(grid = false, legend = false)
gr()

outplotTPTNdetaRiDistrution = plot(layout = layout, link = :both, 
        size = (1800, 1200), margin = (8, :mm), dpi = 300)

histogram!(trainDf_0[:, "DirectMatch"], 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "DirectMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(trainDf_1[:, "DirectMatch"], 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "DirectMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(trainDf_0[:, "ReversMatch"], 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "ReverseMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(trainDf_1[:, "ReversMatch"], 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "ReverseMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(trainDf_DE0[:, "MatchRatio"], 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MatchDifference", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(trainDf_DE1[:, "MatchRatio"], 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MatchDifference", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Training Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(testDf_0[:, "DirectMatch"], 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "DirectMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDf_1[:, "DirectMatch"], 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "DirectMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(testDf_0[:, "ReversMatch"], 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "ReverseMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDf_1[:, "ReversMatch"], 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "ReverseMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(testDf_DE0[:, "MatchRatio"], 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MatchDifference", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(testDf_DE1[:, "MatchRatio"], 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MatchDifference", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Testing Dataset", 
    titlefont = font(12), 
    dpi = 300)



histogram!(pestDf_0[:, "DirectMatch"], 
    subplot = 7, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "DirectMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf_1[:, "DirectMatch"], 
    subplot = 7, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "DirectMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(pestDf_0[:, "ReversMatch"], 
    subplot = 8, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "ReverseMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf_1[:, "ReversMatch"], 
    subplot = 8, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "ReverseMatch", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

histogram!(pestDf_DE0[:, "MatchRatio"], 
    subplot = 9, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MatchDifference", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 0", 
    fc = "pink", 
    lc = "pink", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(pestDf_DE1[:, "MatchRatio"], 
    subplot = 9, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "MatchDifference", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "LABEL 1", 
    fc = "skyblue", 
    lc = "skyblue", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topleft, 
    legendfont = font(8), 
    title = "Preprocessed Independent Dataset", 
    titlefont = font(12), 
    dpi = 300)

# Saving
savefig(outplotTPTNdetaRiDistrution, "F:\\UvA\\outplot_TPTNDistrution_Match.png")

