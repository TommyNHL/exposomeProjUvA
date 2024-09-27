## INPUT(S)
# Database_INTERNAL_2022-11-17.csv
# CNLs_10mDa.csv
# dataAllFP_withNewPredictedRiWithStratification.csv

## OUTPUT(S)
# CNLs_10mDa_missed.csv
# databaseOfInternal_withNLs.csv
# dataframeCNLsRows.csv
# dfCNLsSumModeling.csv
# massesCNLsDistrution.png

## install packages needed ##
#Pkg.add("PyCall")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using CSV, DataFrames

## define a function for data extraction ##
function getVec(matStr)
    if matStr[1] .== '['
        if contains(matStr, ", ")
            str = split(matStr[2:end-1],", ")
        else
            str = split(matStr[2:end-1]," ")
        end
    elseif matStr[1] .== 'A'
        if contains(matStr, ", ")
            str = split(matStr[5:end-1],", ")
        else
            str = split(matStr[5:end-1]," ")
        end
    elseif matStr[1] .== 'F'
        if matStr .== "Float64[]"
            return []
        else
            str = split(matStr[9:end-1],", ")
        end
    elseif matStr[1] .== 'I'
        if matStr .== "Int64[]"
            return []
        else
            str = split(matStr[7:end-1],", ")
        end
    else
        println("New vector start")
        println(matStr)
    end
    if length(str) .== 1 && cmp(str[1],"") .== 0
        return []
    else
        str = parse.(Float64, str)
        return str
    end
end

## input 1095389 x 31 df ##
# column informaiton can refer to Readme
inputDB = CSV.read("F:\\Database_INTERNAL_2022-11-17.csv", DataFrame)

## filter out NEGATIVE ionization mode ## -> 1078844 x 4
## filter out NEGATIVE ionization mode ## -> 817512 x 4
## filter in precusor ion with measured m/z ## -> 817413 x 4
inputData0 = inputDB[inputDB.ION_MODE .!= "NEGATIVE", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES", "ION_MODE"]]
inputData = inputData0[inputData0.ION_MODE .!= "N", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
inputData = inputData[inputData.PRECURSOR_ION .!== NaN, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
inputData = inputData[inputData.PRECURSOR_ION .!= "NaN", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
#= inputData = inputData[inputData.PRECURSOR_ION .<= 1000, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]] =#

## initialize an array for 1 more column ## -> 817413 x 5
inputData[!, "CNLmasses"] .= [[]]
size(inputData)

## calculate neutral loss masses (NLs) ##
## filter in CNL masses-in-interest, and store in Vector{Any}
    ## filter in CNLs features ##
    # according to the pre-defined CNLs in CNLs_10mDa.csv
    # 16022 candidates
    candidates_df = CSV.read("F:\\CNLs_10mDa.csv", DataFrame)
    candidatesList = []
    for can in candidates_df[:, 1]
        push!(candidatesList, round(float(can), digits = 2))
    end
    #
for i in 1:size(inputData, 1)
    println(i)
    fragIons = getVec(inputData[i,"MZ_VALUES"])
    arrNL = Set()
    for frag in fragIons
        if (inputData[i,"PRECURSOR_ION"] - frag) >= float(0)
            NL = round((inputData[i,"PRECURSOR_ION"] - frag), digits = 2)
            if (NL in candidatesList)
                push!(arrNL, NL)
            end
        end
    end
    inputData[i, "CNLmasses"] = sort!(collect(arrNL))
end
sort!(inputData, [:INCHIKEY, :SMILES, :PRECURSOR_ION, :CNLmasses])

## reduce df size (rows) ##
## input 30684 x (2+791) df ##
    # columns include 
    # SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs, 
    # and newly added one (FP-derived predicted Ri)
    inputAllFPDB = CSV.read("F:\\UvA\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
    sort!(inputAllFPDB, [:INCHIKEY, :SMILES])
        
    ## find missing features ##
    function getMasses(db, i, arr)
        massesArr = arr
        masses = db[i, "CNLmasses"]
        for mass in masses
            push!(massesArr, mass)
        end
        return massesArr
    end
        #
        featuresCNLs = []
        for i in 1:size(inputData, 1)
            println(i)
            featuresCNLs = getMasses(inputData, i, featuresCNLs)
        end
        size(featuresCNLs)
        #
        # 27434522 features -> 15994 features
        distinctFeaturesCNLs = Set()
        for featuresCNL in featuresCNLs
            push!(distinctFeaturesCNLs, featuresCNL)
        end
        distinctFeaturesCNLs = sort!(collect(distinctFeaturesCNLs))
        #
        # 16022 candidates -> 15994 candidates
        finalCNLs = []
        whatAreMissed = []
        for candidate in candidatesList
            if (candidate in distinctFeaturesCNLs)
                push!(finalCNLs, candidate)
            else
                push!(whatAreMissed, candidate)
            end
        end
        size(finalCNLs)
        size(whatAreMissed)
        #
    dfMissed = DataFrame([[]], ["whatAreMissed"])
    for miss in whatAreMissed
        list = [miss]
        push!(dfMissed, list)
    end
    #
    dfMissed
    savePath = "F:\\UvA\\CNLs_10mDa_missed.csv"
    CSV.write(savePath, dfMissed)
        
    ## filter in row entries ## according to the presence of FPs in .csv DB
    function haveFPRiOrNot(DB, i)
        if (DB[i, "INCHIKEY"] in Array(inputAllFPDB[:, "INCHIKEY"]) || DB[i, "SMILES"] in Array(inputAllFPDB[:, "SMILES"]))
            return true
        else 
            return false
        end
    end
        ## create a table with 4 columns ##
        dfOutput = DataFrame([[],[],[],[]], ["SMILES", "INCHIKEY", "PRECURSOR_ION", "CNLmasses"])
        size(dfOutput)  # 0 x 4
        #
        # 817413 x 5
        inputData
        dfOutput
        size(inputData[1, "CNLmasses"], 1)
        #
        ref = ""
        for i in 1:size(inputData, 1)
            println(i)
            if ( (size(inputData[i, "CNLmasses"], 1) >= 2) && (haveFPRiOrNot(inputData, i) == true))
                temp = []
                mass = string(inputData[i, "SMILES"], inputData[i, "INCHIKEY"], inputData[i, "PRECURSOR_ION"], inputData[i, "CNLmasses"])
                push!(temp, inputData[i, "SMILES"])
                push!(temp, inputData[i, "INCHIKEY"])
                push!(temp, inputData[i, "PRECURSOR_ION"])
                push!(temp, inputData[i, "CNLmasses"])
                if (mass != ref)
                    push!(dfOutput, temp)
                    ref = mass
                end
            end
        end
    # 693685 x 4
    dfOutput
## save the output table as a spreadsheet ##
# output csv is a 693685 x 4 df
savePath = "F:\\UvA\\databaseOfInternal_withNLs.csv"
CSV.write(savePath, dfOutput)

## transform table as row(ID copounds) x column(CNLs masses) ##
finalFeaturesCNLs = []
for i in 1:size(dfOutput, 1)
    println(i)
    finalFeaturesCNLs = getMasses(dfOutput, i, finalFeaturesCNLs)
end
size(finalFeaturesCNLs)
#
# 25497040 features -> 15961 features
finalDistinctFeaturesCNLs = Set()
for featuresCNL in finalFeaturesCNLs
    push!(finalDistinctFeaturesCNLs, featuresCNL)
end
finalDistinctFeaturesCNLs = sort!(collect(finalDistinctFeaturesCNLs))

## create a table with 4+15961 columns features CNLs ##
finalColumnsCNLs = []
for distinctFeaturesCNL in finalDistinctFeaturesCNLs
    push!(finalColumnsCNLs, string(distinctFeaturesCNL))
end
size(finalColumnsCNLs)

## store data in a Matrix ##
X = zeros(693685, 15961)
#
for i in 1:size(dfOutput, 1)
    println(i)
    arr = []
    arr = getMasses(dfOutput, i, arr)
    mumIon = round(dfOutput[i, "PRECURSOR_ION"], digits = 2)
    for col in arr
        mz = findall(x->x==col, finalDistinctFeaturesCNLs)
        if (col <= mumIon)
            X[i, mz] .= 1
        elseif (col > mumIon)
            X[i, mz] .= -1
        end
    end
end
#
dfCNLs = DataFrame(X, finalColumnsCNLs)
insertcols!(dfCNLs, 1, ("ENTRY"=>collect(1:693685)))
insertcols!(dfCNLs, 2, ("SMILES"=>dfOutput[:, "SMILES"]))
insertcols!(dfCNLs, 3, ("INCHIKEY"=>dfOutput[:, "INCHIKEY"]))
insertcols!(dfCNLs, 4, ("MONOISOTOPICMASS"=>((dfOutput[:, "PRECURSOR_ION"] .- 1.007276)/1000)))
size(dfCNLs)  # 693685 x (3+1+15961)

## ouput df 693685 x (3+1+15961) ##
savePath = "F:\\UvA\\dataframeCNLsRows.csv"
CSV.write(savePath, dfCNLs)
println("done for saving csv")

## plot histogram ##
desStat = describe(dfCNLs)  # 15965 x 7
desStat[5,:]
#
sumUp = []
push!(sumUp, 888888)
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, 888888)
for col in names(dfCNLs)[5:end]
    count = 0
    for i in 1:size(dfCNLs, 1)
        count += dfCNLs[i, col]
    end
    push!(sumUp, count)
end
push!(dfCNLs, sumUp)
# 693685 -> 693686 rows
dfCNLsSum = dfCNLs[end:end,:]  #693686
savePath = "F:\\UvA\\dfCNLsSumModeling.csv"
CSV.write(savePath, dfCNLsSum)
#
using DataSci4Chem
massesCNLsDistrution = bar(finalDistinctFeaturesCNLs, Vector(dfCNLs[end, 5:end]), 
    label = false, 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    xtickfontsize = 12, 
    ytickfontsize= 12, 
    xlabel="Feature CNL mass", xguidefontsize=16, 
    ylabel="Count", yguidefontsize=16, 
    dpi = 300)
## Save figure ##
savefig(massesCNLsDistrution, "F:\\UvA\\massesCNLsDistrution.png")
