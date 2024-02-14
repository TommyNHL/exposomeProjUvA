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

# inputing 1095389 x 31 df
inputDB = CSV.read("D:\\0_data\\Database_INTERNAL_2022-11-17.csv", DataFrame)

# filtering out NEGATIVE ionization mode -> 1078844 x 4
# filtering out NEGATIVE ionization mode -> 834057 x 4
# filtering in precusor ion with measured m/z -> 833958 x 4
inputData = inputDB[inputDB.ION_MODE .!= "NEGATIVE", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
inputData = inputDB[inputDB.ION_MODE .!= "N", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
inputData = inputData[inputData.PRECURSOR_ION .!== NaN, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
#= inputData = inputData[inputData.PRECURSOR_ION .<= 1000, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]] =#

# initialization for 2 more columns -> 833958 x 5
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
# output csv is a 833958 x 5 df
savePath = "D:\\0_data\\databaseOfInternal_withNLs.csv"
CSV.write(savePath, inputData)

# concating with the Cocamide MS2 spectra DB and outputing, 833958 + 52030 = 885988 x 5
savePath = "D:\\0_data\\databaseOfAllMS2_withNLs.csv"
inputDBcocamide = CSV.read("D:\\0_data\\databaseOfMassBankEUMoNAdeletedNA_withNLs.csv", DataFrame)
outputDB = vcat(inputData, inputDBcocamide)
CSV.write(savePath, outputDB)