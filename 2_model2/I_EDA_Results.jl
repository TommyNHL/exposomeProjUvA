VERSION
## install packages needed ##
#using Pkg
#Pkg.add("CairoMakie")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using StatsPlots
using Plots

## import training set ##
trainDfall = CSV.read("F:\\UvA\\results\\CocamideExtWithStartification_Fingerprints_train_predict_err.csv", DataFrame)
trainDfapp = CSV.read("F:\\UvA\\results\\CocamideExtWithStartification_Fingerprints_train_predict_err2.csv", DataFrame)

## import testing set ##
testDfall = CSV.read("F:\\UvA\\results\\CocamideExtWithStartification_Fingerprints_test_predict_err.csv", DataFrame)
testDfapp = CSV.read("F:\\UvA\\results\\CocamideExtWithStartification_Fingerprints_test_predict_err2.csv", DataFrame)

# ==================================================================================================
## plot graph ##
using DataSci4Chem
#
layout = @layout [a{0.50w, 0.7h} b{0.50w, 0.7h} 
                  c{0.50w, 0.3h} d{0.50w, 0.3h} ]
default(grid = false, legend = false)
gr()
#
outplot = plot(layout = layout, link = :both, 
        size = (1200, 900), margin = (8, :mm), dpi = 300)
#
histogram!(trainDfall[:, "RI"], bins = 150, 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "RTI Values", xguidefontsize=12, 
    ylabel = "Count", yguidefontsize=12, 
    label = "True Values (All Calibrants)", 
    fc = "purple", 
    lc = "purple", 
    alpha = 0.25, 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 10, 
    ytickfontsize= 10, 
    legend = :topright, 
    legendfont = font(10), 
    title = "Model 1 Training Dataset", 
    titlefont = font(18), 
    dpi = 300)
    histogram!(trainDfapp[:, "RI"], bins = 150, 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "RTI Values", xguidefontsize=12, 
    ylabel = "Count", yguidefontsize=12, 
    label = "True Values (with Public Spectra)", 
    fc = "blue", 
    lc = "blue", 
    alpha = 0.25, 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 10, 
    ytickfontsize= 10, 
    legend = :topright, 
    legendfont = font(10), 
    title = "Model 1 Training Dataset", 
    titlefont = font(18), 
    dpi = 300)
    histogram!(trainDfapp[:, "MFpredictRI"], bins = 150, 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "RTI Values", xguidefontsize=12, 
    ylabel = "Count", yguidefontsize=12, 
    label = "MF-Derived Values", 
    fc = "skyblue", 
    lc = "black", 
    alpha = 0.75, 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 10, 
    ytickfontsize= 10, 
    legend = :topright, 
    legendfont = font(10), 
    title = "Model 1 Training Dataset", 
    titlefont = font(18), 
    dpi = 300)
    histogram!(trainDfapp[:, "CNLpredictRI"], bins = 150, 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "RTI Values", xguidefontsize=12, 
    ylabel = "Count", yguidefontsize=12, 
    label = "CNL-Derived Values", 
    fc = "green", 
    lc = "green", 
    alpha = 0.5, 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 10, 
    ytickfontsize= 10, 
    legend = :topright, 
    legendfont = font(10), 
    title = "Model 1 Training Dataset", 
    titlefont = font(18), 
    dpi = 300)
histogram!(testDfall[:, "RI"], bins = 150, 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "RTI Values", xguidefontsize=12, 
    ylabel = "Count", yguidefontsize=12, 
    label = "True Values (All Calibrants)", 
    fc = "purple", 
    lc = "purple", 
    alpha = 0.25, 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 10, 
    ytickfontsize= 10, 
    legend = :topright, 
    legendfont = font(10), 
    title = "Model 1 Testing Dataset", 
    titlefont = font(18), 
    dpi = 300)
    histogram!(testDfapp[:, "RI"], bins = 150, 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "RTI Values", xguidefontsize=12, 
    ylabel = "Count", yguidefontsize=12, 
    label = "True Values (with Public Spectra)", 
    fc = "blue", 
    lc = "blue", 
    alpha = 0.25, 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 10, 
    ytickfontsize= 10, 
    legend = :topright, 
    legendfont = font(10), 
    title = "Model 1 Testing Dataset", 
    titlefont = font(18), 
    dpi = 300)
    histogram!(testDfapp[:, "MFpredictRI"], bins = 150, 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "RTI Values", xguidefontsize=12, 
    ylabel = "Count", yguidefontsize=12, 
    label = "MF-Derived Values", 
    fc = "skyblue", 
    lc = "black", 
    alpha = 0.75, 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 10, 
    ytickfontsize= 10, 
    legend = :topright, 
    legendfont = font(10), 
    title = "Model 1 Testing Dataset", 
    titlefont = font(18), 
    dpi = 300)
    histogram!(testDfapp[:, "CNLpredictRI"], bins = 150, 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "RTI Values", xguidefontsize=12, 
    ylabel = "Count", yguidefontsize=12, 
    label = "CNL-Derived Values", 
    fc = "green", 
    lc = "green", 
    alpha = 0.5, 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 10, 
    ytickfontsize= 10, 
    legend = :topright, 
    legendfont = font(10), 
    title = "Model 1 Testing Dataset", 
    titlefont = font(18), 
    dpi = 300)
    plot!(xlims = (-150, 1500), ylims = (0, 50), subplot = 2)
    plot!(xlims = (-150, 1500), ylims = (0, 100), subplot = 1)
## save ##
#savefig(outplot, "F:\\UvA\\results\\outplot_fig2AB.png")
boxplot!(trainDfapp[:, "CNLError"], 
    subplot = 3, 
    orientation = :horizontal, 
    framestyle = :box, 
    seriestype=:stephist, 
    color = "green", 
    fc = "green", 
    lc = "black", 
    alpha = 0.5, 
    lw = 1, 
    yaxis=([], false), 
    margin = (5, :mm), 
    dpi = 300)
boxplot!(trainDfapp[:, "MFError"], 
    subplot = 3, 
    orientation = :horizontal, 
    framestyle = :box, 
    seriestype=:stephist, 
    color = "skyblue", 
    fc = "skyblue", 
    lc = "black", 
    alpha = 0.75, 
    lw = 1, 
    yaxis=([], false), 
    margin = (5, :mm), 
    dpi = 300,
    xlabel = "RTI Error (Predicted Value - True Value)", xguidefontsize=12)

boxplot!(testDfapp[:, "CNLError"], 
    subplot = 4, 
    orientation = :horizontal, 
    framestyle = :box, 
    seriestype=:stephist, 
    color = "green", 
    fc = "green", 
    lc = "black", 
    alpha = 0.5, 
    lw = 1, 
    yaxis=([], false), 
    margin = (5, :mm), 
    dpi = 300)
boxplot!(testDfapp[:, "MFError"], 
    subplot = 4, 
    orientation = :horizontal, 
    framestyle = :box, 
    seriestype=:stephist, 
    color = "skyblue", 
    fc = "skyblue", 
    lc = "black", 
    alpha = 0.75, 
    lw = 1, 
    yaxis=([], false), 
    margin = (5, :mm), 
    dpi = 300, 
    xlabel = "RTI Error (Predicted Value - True Value)", xguidefontsize=12)
    plot!(xlims = (-1000, 1000), ylims = (8.5, 10.5), subplot = 3)
    plot!(xlims = (-1000, 1000), ylims = (10.5, 12.5), subplot = 4)
## save ##
savefig(outplot, "F:\\UvA\\results\\outplot_fig2ABCD.png")
