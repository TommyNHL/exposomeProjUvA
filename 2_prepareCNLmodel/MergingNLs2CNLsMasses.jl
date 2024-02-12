using Pkg
#Pkg.add("BSON")
#Pkg.add(PackageSpec(url=""))
#using BSON
using CSV, DataFrames, PyCall, Conda, LinearAlgebra, Statistics
Conda.add("pubchempy")
Conda.add("padelpy")
Conda.add("joblib")
## import packages ##
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
jl = pyimport("joblib")

# inputing 88121 x 6 df
# columns: SMILES, INCHIKEY, PRECURSOR_ION, MZ_VALUES, binedPRECURSOR_ION, CNLmasses
inputDB = CSV.read("D:\\0_data\\databaseOfAllMS2_withNLs.csv", DataFrame)
sort!(inputDB, :INCHIKEY)

# merging all CNLmasses of 1 compounds from different sources into a list of set
mergedDf = DataFrame([[],[],[],[]], 
    ["SMILES", "INCHIKEY", "binedPRECURSOR_ION", "CNLmasses"])

function getFrags(str)
    masses = split(str, ", ")
end

function getMasses(i, arr)
    massesArr = arr
    masses = getFrags(inputDB[i, "CNLmasses"])
    for mass in masses
        push!(massesArr, Float64[BigFloat(mass)][1])
    end
    return massesArr
end

function newRowOrNot(str1, str2)
    if (string(str1) == string(str2))
        return true
    else
        return false
    end
end

function setArr(arr)
    arrSet = Set()
    for frag in arr
        push!(arrSet, frag)
    end
    return arrSet
end

massesArr = []
count = 1
for i in 1:size(inputDB, 1)
    println(i)
    if (i == 1 && count == 1)
        massesArr = getMasses(i, massesArr)
        #println(massesArr)
    elseif (i > 1 && newRowOrNot(inputDB[i-1, "INCHIKEY"], inputDB[i, "INCHIKEY"]) == true)
        massesArr = getMasses(i, massesArr)
        #println(massesArr)
    elseif (i > 1 && newRowOrNot(inputDB[i-1, "INCHIKEY"], inputDB[i, "INCHIKEY"]) == false)
        append!(mergedDf, DataFrame(
                  SMILES = [inputDB[i, "SMILES"]], 
                  INCHIKEY = [inputDB[i, "INCHIKEY"]], 
                  binedPRECURSOR_ION = [inputDB[i, "binedPRECURSOR_ION"]], 
                  CNLmasses = [sort!(collect(setArr(massesArr)))]
                  ))
        count += 1
        massesArr = []
        massesArr = getMasses(i, massesArr)
        #println(massesArr)
    end
end

mergedDf

# ouputing df 2521 x 4
savePath = "D:\\0_data\\databaseOfAllMS2_withMergedNLs.csv"
CSV.write(savePath, mergedDf)