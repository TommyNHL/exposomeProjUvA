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

#import csv
# input csv is a MS2 DBs of MassBank EU & MoNA, 68677 x 32 df 
# 3 rows deleted due to missing value for ionization mode -> 68674 x 32 df

# inputing 68674 x 32 df
inputDB = CSV.read("D:\\0_data\\databaseOfMassBankEUMoNAdeletedNA.csv", DataFrame)

# filtering in positive ionization mode -> 52129 x 4
# filtering in precusor ion with measured m/z -> 52030 x 4
inputData = inputDB[inputDB.ION_MODE .== "POSITIVE", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
inputData = inputData[inputData.PRECURSOR_ION .!== NaN, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
#= inputData = inputData[inputData.PRECURSOR_ION .<= 1000, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]] =#

# initialization for 2 more columns -> 52030 x 5
inputData[!, "CNLmasses"] .= String[string("")]
size(inputData)

# NLs calculation, Any[] -> String
for i in 1:size(inputData, 1)
    println(i)
    fragIons = getVec(inputData[i,"MZ_VALUES"])
    arrNL = string("")
    for frag in fragIons
        if arrNL == string("")
          NL = round((inputData[i,"PRECURSOR_ION"] - frag), digits = 2)
          arrNL = string(arrNL, string(NL))
        else
          NL = round((inputData[i,"PRECURSOR_ION"] - frag), digits = 2)
          arrNL = string(arrNL, ", ", string(NL))
          #println(arrNL)
        end
    end
    inputData[i, "CNLmasses"] = arrNL
end

# save
# output csv is a 52030 x 5 df
savePath = "D:\\0_data\\databaseOfMassBankEUMoNAdeletedNA_withNLs.csv"
CSV.write(savePath, inputData)