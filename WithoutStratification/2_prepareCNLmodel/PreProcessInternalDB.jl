using Pkg
#Pkg.add("BSON")
#Pkg.add(PackageSpec(url=""))
#using BSON
using CSV, DataFrames #, PyCall, Conda, LinearAlgebra, Statistics
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#pcp = pyimport("pubchempy")
#pd = pyimport("padelpy")
#jl = pyimport("joblib")

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

# inputing 1095389 x 31 df
inputDB = CSV.read("D:\\0_data\\Database_INTERNAL_2022-11-17.csv", DataFrame)

# filtering out NEGATIVE ionization mode -> 1078844 x 4
# filtering out NEGATIVE ionization mode -> 817512 x 4
# filtering in precusor ion with measured m/z -> 817413 x 4
inputData0 = inputDB[inputDB.ION_MODE .!= "NEGATIVE", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES", "ION_MODE"]]
inputData = inputData0[inputData0.ION_MODE .!= "N", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
inputData = inputData[inputData.PRECURSOR_ION .!== NaN, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
#= inputData = inputData[inputData.PRECURSOR_ION .<= 1000, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]] =#

# initialization for 1 more column -> 817413 x 5
inputData[!, "CNLmasses"] .= [[]]
size(inputData)

# NLs calculation, filtering CNL-in-interest, storing in Vector{Any}
        # filtering in CNLs features according to the pre-defined CNLs in CNLs_10mDa.csv
        # inputing 16022 candidates
        candidates_df = CSV.read("D:\\0_data\\CNLs_10mDa.csv", DataFrame)
        candidatesList = []
        for can in candidates_df[:, 1]
            push!(candidatesList, round(float(can), digits = 2))
        end
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

# Reducing df size (rows)
# inputing 30684 x (2+791) df, columns include 
        #SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs, 
        #and newly added one (FP-derived predicted Ri)
        inputAllFPDB = CSV.read("D:\\0_data\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
        sort!(inputAllFPDB, [:INCHIKEY, :SMILES])
        
        # finding missing features
        inputData[1,5]

        test = []
        function getMasses(db, i, arr)
            massesArr = arr
            masses = db[i, "CNLmasses"]
            for mass in masses
                push!(massesArr, mass)
            end
            return massesArr
        end
        
        test = getMasses(inputData, 1, test)
        test
        test = getMasses(inputData, 1, test)
        test
        
        featuresCNLs = []
        for i in 1:size(inputData, 1)
            println(i)
            featuresCNLs = getMasses(inputData, i, featuresCNLs)
        end
        size(featuresCNLs)

        # 27434522 features -> 15994 features
        distinctFeaturesCNLs = Set()
        for featuresCNL in featuresCNLs
            push!(distinctFeaturesCNLs, featuresCNL)
        end
        distinctFeaturesCNLs = sort!(collect(distinctFeaturesCNLs))

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

        dfMissed = DataFrame([[]], ["whatAreMissed"])
        for miss in whatAreMissed
            list = [miss]
            push!(dfMissed, list)
        end

        savePath = "D:\\0_data\\CNLs_10mDa_missed.csv"
        CSV.write(savePath, dfMissed)
        
        # filtering in row entries according to the presence of FPs in .csv DB
        function haveFPRiOrNot(DB, i)
            if (DB[i, "INCHIKEY"] in Array(inputAllFPDB[:, "INCHIKEY"]) || DB[i, "SMILES"] in Array(inputAllFPDB[:, "SMILES"]))
                return true
            else 
                return false
            end
        end

        # creating a table with 4 columns
        dfOutput = DataFrame([[],[],[],[]], ["SMILES", "INCHIKEY", "PRECURSOR_ION", "CNLmasses"])
        size(dfOutput)  # 0 x 4

# 817413 x 5
inputData
dfOutput
size(inputData[1, "CNLmasses"], 1)

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

# save
# output csv is a 693685 x 4 df
savePath = "D:\\0_data\\databaseOfInternal_withNLs.csv"
CSV.write(savePath, dfOutput)

# transform table as row(ID copounds) x column(CNLs masses)
finalFeaturesCNLs = []
for i in 1:size(dfOutput, 1)
    println(i)
    finalFeaturesCNLs = getMasses(dfOutput, i, finalFeaturesCNLs)
end
size(finalFeaturesCNLs)

# 25497040 features -> 15961 features
finalDistinctFeaturesCNLs = Set()
for featuresCNL in finalFeaturesCNLs
    push!(finalDistinctFeaturesCNLs, featuresCNL)
end
finalDistinctFeaturesCNLs = sort!(collect(finalDistinctFeaturesCNLs))

# creating a table with 4+15961 columns features CNLs
finalColumnsCNLs = []
for distinctFeaturesCNL in finalDistinctFeaturesCNLs
    push!(finalColumnsCNLs, string(distinctFeaturesCNL))
end
size(finalColumnsCNLs)

# storing data in a Matrix
X = zeros(693685, 15961)

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
            X[i, mz1] .= -1
        end
    end
end

dfCNLs = DataFrame(X, finalColumnsCNLs)
insertcols!(dfCNLs, 1, ("ENTRY"=>collect(1:693685)))
insertcols!(dfCNLs, 2, ("SMILES"=>dfOutput[:, "SMILES"]))
insertcols!(dfCNLs, 3, ("INCHIKEY"=>dfOutput[:, "INCHIKEY"]))
insertcols!(dfCNLs, 4, ("ISOTOPICMASS"=>dfOutput[:, "PRECURSOR_ION"]-1.007276))
size(dfCNLs)  # 693685 x (3+1+15961)

# ouputing df 693685 x (3+1+15961)
savePath = "D:\\0_data\\dataframeCNLsRows.csv"
CSV.write(savePath, dfCNLs)

desStat = describe(dfCNLs)  # 15965 x 7
desStat[4,:]

sumUp = []
push!(sumUp, 888888)
push!(sumUp, "summation")
push!(sumUp, "summation")
push!(sumUp, "summation")
for col in names(dfCNLs)[4:end]
    count = 0
    for i in 1:size(dfCNLs, 1)
        count += dfCNLs[i, col]
    end
    push!(sumUp, count)
end
push!(dfCNLs, sumUp)
# 693685 -> 693686 rows
dfCNLs[end,:]  #693686

using DataSci4Chem
massesCNLsDistrution = bar(names(dfCNLs)[4:end], Vector(dfCNLs[end, 4:end]), 
    label = false, 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    xlabel!("CNLs features")
    ylabel!("Summation")
    # Saving
    savefig(massesCNLsDistrution, "D:\\2_output\\massesCNLsDistrution.png")