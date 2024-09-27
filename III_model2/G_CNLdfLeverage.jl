## INPUT(S)
# dataframe73_95dfTrainSetWithStratification.csv

## OUTPUT(S)
# dataframe73_dfTrainSetWithStratification_95FPCNLleverage.csv
# CNL model 95% leverage cut-off = 0.14604417882015916
# CNLLeverageValueDistrution.png

VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using CSV, DataFrames, LinearAlgebra, Statistics
using ProgressBars
using Plots, DataFrames

## input ##
# 485577 x 1+1+1+15961+1 df = 485577 x 15965
dfOutputCNL = CSV.read("F:\\UvA\\dataframe73_95dfTrainSetWithStratification.csv", DataFrame)
sort!(dfOutputCNL, [:ENTRY])

## copy ##
X = deepcopy(dfOutputCNL[:, 3:end-1])  # 485577 x 790 df
size(X)
Y = deepcopy(dfOutputCNL[:, end])  # 485577,
size(Y)
Xmat = Matrix(X)

## compute leverage ##
# 790 x 790
hipinv = zeros(15962, 15962)
hipinv[:,:] .= pinv(Xmat'*Xmat)
    ## define a function ##
    function leverage_dist(X)   # Set x1 and x2 to your FPs variables
        h = zeros(485577,1)
        for i in ProgressBar(1: size(X,1)) #check dimensions
            x = X[i,:]
            #hi = x'*pinv(X'*X)*x
            hi = x'* hipinv *x
            #push!(h,hi)
            h[i,1] = hi
        end
        return h
    end
    #
    h = leverage_dist(Matrix(X))
    ht = Vector(transpose(h)[1,:])
    dfOutputCNL[!, "Leverage"] .= ht
    #
    # 346842-2 x 1
    savePath = "F:\\UvA\\dataframe73_dfTrainSetWithStratification_95FPCNLleverage.csv"
    CSV.write(savePath, dfOutputCNL)
    #
sortHt = ht
sort!(sortHt)
sortHt[end-24279]
finalDistinctLeverage = Set()
lvCount = []
for lv in sortHt
    push!(finalDistinctLeverage, lv)
end
finalDistinctLeverage = sort!(collect(finalDistinctLeverage))
# CNL model 95% leverage cut-off = 0.14604417882015916

## plot graph ##
using DataSci4Chem
leverageDistrution = histogram(Vector(sortHt), 
    label = false, 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    xtickfontsize = 12, 
    ytickfontsize= 12, 
    xlabel="Leverage value", xguidefontsize=16, 
    ylabel="Count", yguidefontsize=16, 
    dpi = 300)
    new_xticks = ([0.14604417882015916], ["\$\\bar"])
    vline!(new_xticks[1], label = "95% cuf-off: 0.146", legendfont = font(12), lc = "purple")
    # Saving
    savefig(leverageDistrution, "F:\\UvA\\CNLLeverageValueDistrution.png")
