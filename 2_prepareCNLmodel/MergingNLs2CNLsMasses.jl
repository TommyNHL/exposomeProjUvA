using Pkg
#Pkg.add("BSON")
#Pkg.add(PackageSpec(url=""))
#using BSON
using CSV, DataFrames, PyCall, Conda, LinearAlgebra, Statistics
#Conda.add("pubchempy")
#Conda.add("padelpy")
#Conda.add("joblib")
## import packages ##
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
jl = pyimport("joblib")

# inputing 880403 x 6 df
# columns: SMILES, INCHIKEY, PRECURSOR_ION, MZ_VALUES, binedPRECURSOR_ION, CNLmasses
inputDB = CSV.read("D:\\0_data\\databaseOfAllMS2_withNLs.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES, :PRECURSOR_ION])

# merging all CNLmasses of 1 compounds from different sources into a list of set
# while filtering out NA or N/A IDs
mergedDf = DataFrame([[],[],[],[]], 
    ["SMILES", "INCHIKEY", "CNLmasses", "PRECURSOR_ION"])

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

function determineNA(i)
    return (
        ("NA" in inputDB[i:i,"SMILES"] || "N/A" in inputDB[i:i,"SMILES"]) == true
        ) && (
            ("NA" in inputDB[i:i,"INCHIKEY"] || "N/A" in inputDB[i:i,"INCHIKEY"]) == true
            )
end

function determineInChiKey(i)
    return ("NA" in inputDB[i:i,"INCHIKEY"] || "N/A" in inputDB[i:i,"INCHIKEY"])
end

massesArr = []
for i in 1:size(inputDB, 1)
    println(i)
    if (i == 1 && determineNA(i) == false)
        massesArr = getMasses(i, massesArr)
        #println(massesArr)
    elseif (i == size(inputDB, 1) && newRowOrNot(inputDB[i-1, "INCHIKEY"], inputDB[i, "INCHIKEY"]) == true && determineInChiKey(i) == false)
        if (determineNA(i) == false)
            massesArr = getMasses(i, massesArr)
            append!(mergedDf, DataFrame(
                    SMILES = [inputDB[i, "SMILES"]], 
                    INCHIKEY = [inputDB[i, "INCHIKEY"]], 
                    PRECURSOR_ION = [inputDB[i, "PRECURSOR_ION"]], 
                    CNLmasses = [sort!(collect(setArr(massesArr)))]
                    ))
        end
    elseif (i == size(inputDB, 1) && newRowOrNot(inputDB[i-1, "INCHIKEY"], inputDB[i, "INCHIKEY"]) == true && determineInChiKey(i) == true)
        if (determineNA(i) == false && newRowOrNot(inputDB[i-1, "SMILES"], inputDB[i, "SMILES"]) == true)
            massesArr = getMasses(i, massesArr)
            append!(mergedDf, DataFrame(
                    SMILES = [inputDB[i, "SMILES"]], 
                    INCHIKEY = [inputDB[i, "INCHIKEY"]], 
                    PRECURSOR_ION = [inputDB[i, "PRECURSOR_ION"]], 
                    CNLmasses = [sort!(collect(setArr(massesArr)))]
                    ))
        elseif (determineNA(i) == false && newRowOrNot(inputDB[i-1, "SMILES"], inputDB[i, "SMILES"]) == false)
            if (determineNA(i-1) == false)
                append!(mergedDf, DataFrame(
                    SMILES = [inputDB[i-1, "SMILES"]], 
                    INCHIKEY = [inputDB[i-1, "INCHIKEY"]], 
                    PRECURSOR_ION = [inputDB[i, "PRECURSOR_ION"]], 
                    CNLmasses = [sort!(collect(setArr(massesArr)))]
                    ))
            end
            massesArr = []
            massesArr = getMasses(i, massesArr)
            append!(mergedDf, DataFrame(
                    SMILES = [inputDB[i, "SMILES"]], 
                    INCHIKEY = [inputDB[i, "INCHIKEY"]], 
                    PRECURSOR_ION = [inputDB[i, "PRECURSOR_ION"]], 
                    CNLmasses = [sort!(collect(setArr(massesArr)))]
                    ))
        end
    elseif (i == size(inputDB, 1) && newRowOrNot(inputDB[i-1, "INCHIKEY"], inputDB[i, "INCHIKEY"]) == false)
        if (determineNA(i-1) == false)
            append!(mergedDf, DataFrame(
                        SMILES = [inputDB[i-1, "SMILES"]], 
                        INCHIKEY = [inputDB[i-1, "INCHIKEY"]], 
                        PRECURSOR_ION = [inputDB[i, "PRECURSOR_ION"]], 
                        CNLmasses = [sort!(collect(setArr(massesArr)))]
                        ))
        end
        if (determineNA(i) == false)
            massesArr = []
            massesArr = getMasses(i, massesArr)
            append!(mergedDf, DataFrame(
                    SMILES = [inputDB[i, "SMILES"]], 
                    INCHIKEY = [inputDB[i, "INCHIKEY"]], 
                    PRECURSOR_ION = [inputDB[i, "PRECURSOR_ION"]], 
                    CNLmasses = [sort!(collect(setArr(massesArr)))]
                    ))
        end
    elseif (i > 1 && newRowOrNot(inputDB[i-1, "INCHIKEY"], inputDB[i, "INCHIKEY"]) == true && determineInChiKey(i) == false)
        if (determineNA(i) == false)
            massesArr = getMasses(i, massesArr)
            #println(massesArr)
        end
    elseif (i > 1 && newRowOrNot(inputDB[i-1, "INCHIKEY"], inputDB[i, "INCHIKEY"]) == true && determineInChiKey(i) == true)
        if (determineNA(i) == false && newRowOrNot(inputDB[i-1, "SMILES"], inputDB[i, "SMILES"]) == true)
            massesArr = getMasses(i, massesArr)
            #println(massesArr)
        elseif (determineNA(i) == false && newRowOrNot(inputDB[i-1, "SMILES"], inputDB[i, "SMILES"]) == false)
            if (determineNA(i-1) == false)
                append!(mergedDf, DataFrame(
                        SMILES = [inputDB[i-1, "SMILES"]], 
                        INCHIKEY = [inputDB[i-1, "INCHIKEY"]], 
                        PRECURSOR_ION = [inputDB[i, "PRECURSOR_ION"]], 
                        CNLmasses = [sort!(collect(setArr(massesArr)))]
                        ))
            end
            massesArr = []
            massesArr = getMasses(i, massesArr)
            #println(massesArr)
        elseif (determineNA(i) == true && newRowOrNot(inputDB[i-1, "SMILES"], inputDB[i, "SMILES"]) == false)
            if (determineNA(i-1) == false)
                append!(mergedDf, DataFrame(
                        SMILES = [inputDB[i-1, "SMILES"]], 
                        INCHIKEY = [inputDB[i-1, "INCHIKEY"]], 
                        PRECURSOR_ION = [inputDB[i, "PRECURSOR_ION"]], 
                        CNLmasses = [sort!(collect(setArr(massesArr)))]
                        ))
            end
            massesArr = []
        end
    elseif (i > 1 && newRowOrNot(inputDB[i-1, "INCHIKEY"], inputDB[i, "INCHIKEY"]) == false)
        if (determineNA(i-1) == false)
            append!(mergedDf, DataFrame(
                    SMILES = [inputDB[i-1, "SMILES"]], 
                    INCHIKEY = [inputDB[i-1, "INCHIKEY"]], 
                    PRECURSOR_ION = [inputDB[i, "PRECURSOR_ION"]], 
                    CNLmasses = [sort!(collect(setArr(massesArr)))]
                    ))
        end
        massesArr = []
        if (determineNA(i) == false)
            massesArr = getMasses(i, massesArr)
            #println(massesArr)
        end
    end
end

mergedDf

# ouputing df 28181 x 4
savePath = "D:\\0_data\\databaseOfAllMS2_withMergedNLs.csv"
CSV.write(savePath, mergedDf)