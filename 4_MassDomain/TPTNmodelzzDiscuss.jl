VERSION
using Pkg
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using ScikitLearn
using StatsPlots
using Plots

# 4 + 8 + 1 + 2 + 1
outputDf = CSV.read("F:\\dataframeTPTNModeling.csv", DataFrame)

outputDf[!, "FPinchikeyrealRi"] .= float(0)
outputDf[!, "P1P0RiError"] .= float(0)
describe(outputDf)[end-5:end, :]

# creating a 4368902 x 3+8+2+1+1 df, 
    ## columns: INCHIKEY_ID, INCHIKEYreal, 8+1 ULSA features, LABEL
    ##                      FP->Ri, CNL->Ri ^
# matching INCHIKEY, 30684 x 793 df
inputFP2Ri = CSV.read("F:\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
sort!(inputFP2Ri, [:INCHIKEY, :SMILES])

for i in 1:size(outputDf, 1)
    println(i)
    if outputDf[i, "INCHIKEYreal"] in Array(inputFP2Ri[:, "INCHIKEY"])
        rowNo = findall(inputFP2Ri.INCHIKEY .== outputDf[i, "INCHIKEYreal"])
        outputDf[i, "FPinchikeyrealRi"] = inputFP2Ri[rowNo[end:end], "predictRi"][1]
        outputDf[i, "P1P0RiError"] = outputDf[i, "CNLpredictRi"] - inputFP2Ri[rowNo[end:end], "predictRi"][1]
    else
        outputDf[i, "FPinchikeyrealRi"] = float(8888888)
        outputDf[i, "P1P0RiError"] = float(8888888)
    end
end
outputDf = outputDf[outputDf.FPinchikeyrealRi .!= float(8888888), :]

# 4086824 x 4 + 8 + 1 + 2 + 1 + 2
outputDf
describe(outputDf)

savePath = "F:\\dataframe_P1P0RiError.csv"
CSV.write(savePath, outputDf)

# 4086824 x 4 + 8 + 1 + 2 + 1 + 2
outputDf = CSV.read("F:\\dataframe_P1P0RiError.csv", DataFrame)

outputDf[!, "P1P0RiErrorRound"] .= Integer(0)

ind_1 = []
ind_0 = []
for i in 1:size(outputDf, 1)
    outputDf[i, "P1P0RiErrorRound"] = Integer(round(float(outputDf[i, "P1P0RiError"]), digits = 0))
    outputDf[i, "DeltaRi"] = Integer(round(float(outputDf[i, "DeltaRi"] * 1000), digits = 0))
    if (outputDf[i, "LABEL"] == 0)
        push!(ind_0, i)
    else
        push!(ind_1, i)
    end
end

using DataSci4Chem
outplotTPTNdetaRiDistrution = histogram(outputDf[ind_1, "LABEL"], outputDf[ind_1, "DeltaRi"], 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "delta Ri", 
    ylabel = "Count", 
    label = "LABEL 1", 
    legend = :topright, 
    margin = (8, :mm), 
    size = (800,600), 
    dpi = 300)
histogram!(outputDf[ind_0, "LABEL"], outputDf[ind_0, "DeltaRi"], 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "delta Ri", 
    ylabel = "Count", 
    label = "LABEL 0", 
    legend = :topright, 
    margin = (8, :mm), 
    size = (800,600), 
    dpi = 300)
    # Saving
    savefig(outplotTPTNdetaRiDistrution, "F:\\outplotTPTNdetaRiDistrution.png")
