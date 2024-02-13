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

# inputing 2522 x 4 df
# columns: SMILES, INCHIKEY, binedPRECURSOR_ION, CNLmasses
inputDB = CSV.read("D:\\0_data\\databaseOfAllMS2_withMergedNLs.csv", DataFrame)
sort!(inputDB, :INCHIKEY)

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

# 21010 features -> 20494 features
distinctFeaturesCNLs = Set()
for featuresCNL in featuresCNLs
    if (featuresCNL > -0.005)
        push!(distinctFeaturesCNLs, featuresCNL)
    end
end
distinctFeaturesCNLs = sort!(collect(distinctFeaturesCNLs))

# creating a table with 20493 columns features with CNLs masses > 0 Da
columnsCNLs = []
for distinctFeaturesCNL in distinctFeaturesCNLs
    push!(columnsCNLs, string(distinctFeaturesCNL))
end
size(distinctFeaturesCNLs)

dfCNLs = DataFrame([[],[]], ["SMILES", "INCHIKEY"])
for col in columnsCNLs
    dfCNLs[:, col] = []
end
size(dfCNLs)  # 0 x (2+20494)

# filling table without tolerance
function dfFilling1or0(i, columnsCNLs)
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
            && Float64[BigFloat(col)][1] <= inputDB[i, "binedPRECURSOR_ION"])
            push!(tempTP, 2)
        elseif (Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"]) 
            && Float64[BigFloat(col)][1] > inputDB[i, "binedPRECURSOR_ION"])
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
            && Float64[BigFloat(col)][1] <= inputDB[i, "binedPRECURSOR_ION"])
            push!(tempTN, 1)
        elseif (Float64[BigFloat(col)][1] in getVec(inputDB[i, "CNLmasses"]) 
            && Float64[BigFloat(col)][1] > inputDB[i, "binedPRECURSOR_ION"])
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
end

dfCNLs
for i in 1:size(inputDB, 1)
    println(i)
    append!(dfCNLs, dfFilling1or0(i, columnsCNLs))
end
dfCNLs

# ouputing df 5042 x 20496
savePath = "D:\\0_data\\dataframeCNLsTPTN_withoutTolerance.csv"
CSV.write(savePath, dfCNLs)

### dummy ###
dfCNLs = CSV.read("D:\\0_data\\dataframeCNLsTPTN.csv", DataFrame)
#############

desStat = describe(dfCNLs)  # 20496 x 7
desStat[3,:]

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
# 5042 -> 5043 rows
dfCNLs[5043,:]

# bar plot for the distribution
using DataSci4Chem
names(dfCNLs)[3:end]
Vector(dfCNLs[end, 3:end])
massesCNLsDistrution = bar(names(dfCNLs)[3:end], Vector(dfCNLs[end, 3:end]), 
    label = false,
    margin = (5, :mm), 
    size = (1000,800), 
    dpi = 300)
    xlabel!("CNLs features")
    ylabel!("Summation")
    # Saving
    savefig(massesCNLsDistrution, "D:\\2_output\\massesCNLsDistrution.png")