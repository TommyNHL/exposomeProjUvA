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
# input csv is a MS2 DBs of MassBank EU & MoNA, 68677 x 32 df, columns include 
    #21211 deleted due to "DATA_TYPE != "EXP" -> 47466 x 32 df
        #ACCESSION, e.g. MSBNK-AAFC-AC000001
        #INSTRUMENT, e.g. Q-Exactive Orbitrap Thermo Scientific
        #INSTRUMENT_TYPE, e.g. LC-ESI-ITFT
        #NAME, e.g. Mellein --- Ochracin --- 8-hydroxy-3-methyl-3,4-dihydroisochromen-1-one
        #FORMULA, e.g. C10H10O3

        #CLASS1, e.g. Natural Product
        #CLASS2, e.g. Fungal metabolite
      #8#INCHIKEY, e.g. KWILGNNWGSNMPA-UHFFFAOYSA-N
        #INCHI, e.g. InChI=1S/C10H10O3/c1-6-5-7-3-2-4-8(11)9(7)10(12)13-6/h2-4,6,11H,5H2,1H3
      #10#SMILES, e.g. CC1CC2=C(C(=CC=C2)O)C(=O)O1

        #PUBCHEM, e.g. CID:28516
      #12#MS_TYPE, e.g. MS2
      #13#ION_MODE, e.g. POSITIVE
        #IONIZATION, e.g. ESI
        #FRAGMENTATION_MODE, e.g. HCD

        #COLLISION_ENERGY, e.g. 10(NCE)
        #IONIZATION_VOLTAGE, e.g. 3.9 kV
        #RESOLUTION, e.g. 17500
        #EXACT_MASS, e.g. 178.06299
      #20#PRECURSOR_ION, e.g. 179.0697

        #PRECURSOR_TYPE, e.g. [M+H]+
      #22#MZ_VALUES, e.g. Any[133.0648, 151.0754, 155.9743, 161.0597, 179.0703]
      #23#MZ_INT, e.g. Any[21905.33203125, 9239.8974609375, 10980.8896484375, 96508.4375, 72563.875]
        #MZ_INT_REL, e.g. Any[]
      #25#DATA_TYPE, e.g. EXP

      #26#DataBaseOrigin, e.g. MassBank EU
      #27#RI_UoA, e.g. 369.9835065
        #AD_UoA_model, e.g. 0.000464624
        #AD_UoA_full, e.g. 6.060862153
      #30#RI_Amide, e.g. 645.8675707

        #AD_Amide_model, e.g. 0.006101892
        #AD_Amide_full, e.g. 5.81049124

# inputing 47466 x 32 df
inputDB = CSV.read("D:\\0_data\\databaseOfMassBankEUMoNA.csv", DataFrame)

# filtering in positive ionization mode -> 36372 x 4
# filtering in precusor ion with measured m/z -> 36355 x 4
# filtering in precusor ion with m/z <= 1000 -> 36250 x 4
inputData = inputDB[inputDB.ION_MODE .== "POSITIVE", 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
inputData = inputData[inputData.PRECURSOR_ION .!== NaN, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]
inputData = inputData[inputData.PRECURSOR_ION .<= 1000, 
    ["SMILES", "INCHIKEY", "PRECURSOR_ION", "MZ_VALUES"]]

inputData[!, "CNLmasses"] .= String[string("")]
size(inputData)

#arrNL_col = String[]
for i in 1:size(inputData,1)
    println(i)
    fragIons = getVec(inputData[i,"MZ_VALUES"])
    arrNL = string("")
    for frag in fragIons
        if arrNL == string("")
          NL = inputData[i,"PRECURSOR_ION"] - frag
          arrNL = string(arrNL, string(NL))
        else
          NL = inputData[i,"PRECURSOR_ION"] - frag
          arrNL = string(arrNL, ", ", string(NL))
          #println(arrNL)
        end
    end
    #append!(arrNL_col, arrNL)
    inputData[i,"CNLmasses"] = arrNL
end

# save
# output csv is a 36250 x 5 df
savePath = "D:\\0_data\\databaseOfMassBankEUMoNA_withNLs.csv"
CSV.write(savePath, inputData)