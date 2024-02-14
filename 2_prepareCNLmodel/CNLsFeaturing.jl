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

# inputing 28302 x 4 df
# columns: SMILES, INCHIKEY, CNLmasses, PRECURSOR_ION
inputDB = CSV.read("D:\\0_data\\databaseOfAllMS2_withMergedNLs.csv", DataFrame)
sort!(inputDB, [:INCHIKEY, :SMILES, :PRECURSOR_ION])

# filtering in CNLs features according to the pre-defined CNLs in CNLs_10mDa.csv
# inputing 16022 candidates
candidatesList = Array(CSV.read("D:\\0_data\\CNLs_10mDa.csv", DataFrame)[:,1])

#inputDB[1,4]
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
#test = getVec(inputDB[1,4])

# defining features without tolerance
featuresCNLs = []
for i in 1:size(inputDB, 1)
    println(i)
    candidates = getVec(inputDB[i, "CNLmasses"])
    for massesCNLs in candidates
        push!(featuresCNLs, massesCNLs)
    end
end
size(featuresCNLs)

# 5863157 features -> 90112 features
distinctFeaturesCNLs = Set()
for featuresCNL in featuresCNLs
    if (featuresCNL >= -0.005)
        push!(distinctFeaturesCNLs, featuresCNL)
    end
end
distinctFeaturesCNLs = sort!(collect(distinctFeaturesCNLs))

# 16022 candidates -> 15977 candidates
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

# creating a table with 90112 columns features with CNLs masses > 0 Da
columnsCNLs = []
#for distinctFeaturesCNL in distinctFeaturesCNLs
for distinctFeaturesCNL in finalCNLs
    push!(columnsCNLs, string(distinctFeaturesCNL))
end
size(columnsCNLs)

dfCNLs = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
for col in columnsCNLs
    dfCNLs[:, col] = []
end
size(dfCNLs)  # 0 x (2+15977)

# filling table
#= function dfTFTNFilling1or0(i, columnsCNLs)
    ## TP
    df = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
    df2 = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
    for col in columnsCNLs
        df[:, col] = []
        df2[:, col] = []
    end
    tempTP = []
    push!(tempTP, inputDB[i, "SMILES"])
    push!(tempTP, inputDB[i, "INCHIKEY"])
    for col in columnsCNLs
        if (Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"]) 
            && Float64[BigFloat(col)][1] <= inputDB[i, "PRECURSOR_ION"])
            push!(tempTP, 2)
        elseif (Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"]) 
            && Float64[BigFloat(col)][1] > inputDB[i, "PRECURSOR_ION"])
            push!(tempTP, -2)
        elseif (!(Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"])))
            push!(tempTP, 0)
        else
            push!(tempTP, 1)
        end
    end
    push!(df, tempTP)
    ## TN
    tempTN = []
    push!(tempTN, inputDB[i, "SMILES"])
    push!(tempTN, inputDB[i, "INCHIKEY"])
    for col in columnsCNLs
        if (Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"]) 
            && Float64[BigFloat(col)][1] <= inputDB[i, "PRECURSOR_ION"])
            push!(tempTN, 1)
        elseif (Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"]) 
            && Float64[BigFloat(col)][1] > inputDB[i, "PRECURSOR_ION"])
            push!(tempTN, -2)
        elseif (!(Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"])))
            push!(tempTN, 0)
        else
            push!(tempTN, 2)
        end
    end
    push!(df2, tempTN)
    append!(df, df2)
    return df
end =#

function df1RowFilling1or0(i, columnsCNLs)
    ## 1 row
    df = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
    for col in columnsCNLs
        df[:, col] = []
    end
    temp = []
    push!(temp, inputDB[i, "SMILES"])
    push!(temp, inputDB[i, "INCHIKEY"])
    for col in columnsCNLs
        if (Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"]) && Float64[BigFloat(col)][1] <= inputDB[i, "PRECURSOR_ION"])
            push!(temp, 1)
        elseif (Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"]) && Float64[BigFloat(col)][1] > inputDB[i, "PRECURSOR_ION"])
            push!(temp, -1)
        elseif (!(Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"])))
            push!(temp, 0)
        end
    end
    push!(df, temp)
    return df
end

dfCNLs
for i in size(inputDB, 1)-35:size(inputDB, 1)
    println(i)
    append!(dfCNLs, df1RowFilling1or0(i, columnsCNLs))
end
dfCNLs

# ouputing df 28302 x (2+15977)
savePath = "D:\\0_data\\dataframeCNLsRows.csv"
CSV.write(savePath, dfCNLs)

desStat = describe(dfCNLs)  # 15979 x 7
desStat[3,:]

#= sumUpTP = []
sumUpTN = []
push!(sumUpTP, "summationTP")
push!(sumUpTP, "summationTP")
push!(sumUpTN, "summationTN")
push!(sumUpTN, "summationTN")
for col in names(dfCNLs)[3:end]
    countTP = 0
    countTN = 0
    for i in 1:size(dfCNLs, 1)
        if (i % 2 == 1)
            countTP += dfCNLs[i, col]
        elseif (i % 2 == 0)
            countTN += dfCNLs[i, col]
        end
    end
    push!(sumUpTP, countTP)
    push!(sumUpTN, countTN)
end
push!(dfCNLs, sumUpTP)
push!(dfCNLs, sumUpTN)
# 5042 -> 5044 rows
dfCNLs[5043,:]
dfCNLs[5044,:] =#

# bar plot for the distribution
#= using DataSci4Chem
#names(dfCNLs)[3:end]
#Vector(dfCNLs[end, 3:end])
massesCNLsDistrution = bar(names(dfCNLs)[3:end], Vector(dfCNLs[end-1, 3:end]), 
    label = "TP", 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    xlabel!("CNLs features")
    ylabel!("Summation")
bar!(names(dfCNLs)[3:end], Vector(dfCNLs[end, 3:end]), 
    label = "TN", 
    lc = "orange", 
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    # Saving
    savefig(massesCNLsDistrution, "D:\\2_output\\massesCNLsDistrution.png") =#

sumUp = []
push!(sumUp, "summation")
push!(sumUp, "summation")
for col in names(dfCNLs)[3:end]
    count = 0
    for i in 1:size(dfCNLs, 1)
        count += dfCNLs[i, col]
    end
    push!(sumUp, count)
end
push!(dfCNLs, sumUp)
# 28302 -> 28303 rows
dfCNLs[35,:]

using DataSci4Chem
massesCNLsDistrution = bar(names(dfCNLs)[3:end], Vector(dfCNLs[end-1, 3:end]), 
    label = false, 
    lc = "skyblue", 
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    xlabel!("CNLs features")
    ylabel!("Summation")
    # Saving
    savefig(massesCNLsDistrution, "D:\\2_output\\massesCNLsDistrution.png")