using Pkg
#Pkg.add("BSON")
#Pkg.add(PackageSpec(url=""))
#using BSON
using CSV, DataFrames  #, PyCall, Conda, LinearAlgebra, Statistics
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
#pcp = pyimport("pubchempy")
#pd = pyimport("padelpy")
#jl = pyimport("joblib")

# inputing 833958 x 5 df
# columns: SMILES, INCHIKEY, PRECURSOR_ION, MZ_VALUES, CNLmasses
inputDB = CSV.read("D:\\0_data\\databaseOfInternal_withNLs.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES, :PRECURSOR_ION])

# imputing 30684 x (2+791) df, columns include 
        #SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs, 
        #and newly added one (FP-derived predicted Ri)
inputAllFPDB = CSV.read("D:\\0_data\\dataAllFP_withNewPredictedRi.csv", DataFrame)
sort!(inputAllFPDB, [:INCHIKEY, :SMILES])

# filtering in CNLs features according to the pre-defined CNLs in CNLs_10mDa.csv
# inputing 16022 candidates
candidatesList = Array(CSV.read("D:\\0_data\\CNLs_10mDa.csv", DataFrame)[:,1])

inputDB[1,5]

function getFrags(str)
    masses = split(str, ", ")
end

arr = []
function getMasses(i, arr)
    massesArr = arr
    masses = getFrags(inputDB[i, "CNLmasses"])
    for mass in masses
        push!(massesArr, Float64[BigFloat(mass)][1])
    end
    return massesArr
end

test = getMasses(1, arr)
test

# defining features
featuresCNLs = []
for i in 1:size(inputDB, 1)
    println(i)
    candidates = getMasses(i, featuresCNLs)
end
size(featuresCNLs)

# 31829787 features -> 95633 features
distinctFeaturesCNLs = Set()
for featuresCNL in featuresCNLs
    if (featuresCNL >= -0.005)
        push!(distinctFeaturesCNLs, featuresCNL)
    end
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

# creating a table with 3+15994 columns features CNLs
columnsCNLs = []
for distinctFeaturesCNL in finalCNLs
    push!(columnsCNLs, string(distinctFeaturesCNL))
end
size(columnsCNLs)

dfCNLs = DataFrame([[],[],[]], ["ENTRY", "SMILES", "INCHIKEY"])
for col in columnsCNLs
    dfCNLs[:, col] = []
end
size(dfCNLs)  # 0 x (3+15994)

function df1RowFilling1or0(count, i, columnsCNLs)
    ## 1 row
    temp = []
    push!(temp, count)
    push!(temp, inputDB[i, "SMILES"])
    push!(temp, inputDB[i, "INCHIKEY"])
    for col in finalCNLs
        arr = []
        arr = getMasses(i, arr)
        mumIon = inputDB[i, "PRECURSOR_ION"]
        if (col in arr && col <= mumIon)
            push!(temp, 1)
        elseif (col in arr && col > mumIon)
            push!(temp, -1)
        else
            push!(temp, 0)
        end
    end
    return temp
end

#= function determineNA(i)
    return (
        ("NA" in inputDB[i:i,"SMILES"] || "N/A" in inputDB[i:i,"SMILES"]) == true
        ) && (
            ("NA" in inputDB[i:i,"INCHIKEY"] || "N/A" in inputDB[i:i,"INCHIKEY"]) == true
            )
end =#

dfCNLs

count = 0
for i in 1:5000 #size(inputDB, 1)  #2000
    println(i)
    if (haveFPRiOrNot(inputDB, i) == true)
        count += 1
        push!(dfCNLs, df1RowFilling1or0(count, i, columnsCNLs))
    end
end

dfCNLs

# ouputing df 28302 x (3+15994)
savePath = "D:\\0_data\\dataframeCNLsRows.csv"
CSV.write(savePath, dfCNLs)

desStat = describe(dfCNLs)  # 15979 x 7
desStat[4,:]

sumUp = []
push!(sumUp, 888888)
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
# 28302 -> 28303 rows
dfCNLs[5001,:]  #5001

using DataSci4Chem
massesCNLsDistrution = bar(names(dfCNLs)[4:end], Vector(dfCNLs[5001, 4:end]),  #1000
    label = false, 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    xlabel!("CNLs features")
    ylabel!("Summation")
    # Saving
    savefig(massesCNLsDistrution, "D:\\2_output\\massesCNLsDistrution.png")