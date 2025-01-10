using CSV, DataFrames

## define functions
    function id2RI(df, i)
      ID = df[i, "SMILES"]
      idx = findall(inputAllFPDB.SMILES .== ID)
      if (size(idx, 1) == 0)
          return float(0)
      end
      return inputAllFPDB[idx[end:end], "predictRi"][1]
    end

## import csv #
    # 30684 x 793 df, columns include 
        #SMILES, INCHIKEY, 780 APC2D FPs, 10 Pubchem converted FPs, Predicted Expected RI
        inputAllFPDB = CSV.read("F:\\UvA\\F\\UvA\\dataAllFP_withNewPredictedRiWithStratification.csv", DataFrame)
        sort!(inputAllFPDB, [:INCHIKEY, :SMILES])

    # 5048 x 932 df 
    inputCocamidesTrain = CSV.read("F:\\UvA\\F\\UvA\\dataframe73_dfTrainSetWithStratification_withCNLPredictedRi.csv", DataFrame)
    inputCocamidesTrain = inputCocamidesTrain[:, vcat(1,3)]
    sort!(inputCocamidesTrain, [:SMILES])
    inputCocamidesTrain[!, "dataset"] .= "train"
    inputCocamidesTrain[!, "predictRI"] .= float(0)
    inputCocamidesTrain[!, "Error"] .= float(0)
    for i in 1:size(inputCocamidesTrain, 1)
        inputCocamidesTrain[i, "predictRI"] = id2RI(inputCocamidesTrain, i)
        inputCocamidesTrain[i, "Error"] = inputCocamidesTrain[i, "predictRI"] - inputCocamidesTrain[i, "RI"]
    end
    inputCocamidesTrain = inputCocamidesTrain[inputCocamidesTrain.predictRI .!= float(0), :]
    sort!(inputCocamidesTrain, [:Error])
    print(inputCocamidesTrain[1, "Error"])  # -926.4
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.25), "Error"])  # -81.1
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.5), "Error"])  # 11.7
    print(inputCocamidesTrain[floor(Int, size(inputCocamidesTrain, 1)*0.75), "Error"])  # 69.3
    print(inputCocamidesTrain[size(inputCocamidesTrain, 1)*1, "Error"])  # 390.6
    describe(inputCocamidesTrain)
    # avg = -11.2

    # 1263 x 932 df
    inputCocamidesTest = CSV.read("F:\\UvA\\0_scriptsChemicalSpace\\dataframe73_dfTestSetWithStratification_withCNLPredictedRi.csv", DataFrame)
    inputCocamidesTest = inputCocamidesTest[:, vcat(1,3)]
    sort!(inputCocamidesTest, [:SMILES])
    inputCocamidesTest[!, "dataset"] .= "test"
    inputCocamidesTest[!, "predictRI"] .= float(0)
    inputCocamidesTest[!, "Error"] .= float(0)
    for i in 1:size(inputCocamidesTest, 1)
        inputCocamidesTest[i, "predictRI"] = id2RI(inputCocamidesTest, i)
        inputCocamidesTest[i, "Error"] = inputCocamidesTest[i, "predictRI"] - inputCocamidesTest[i, "RI"]
    end
    inputCocamidesTest = inputCocamidesTest[inputCocamidesTest.predictRI .!= float(0), :]
    sort!(inputCocamidesTest, [:Error])
    print(inputCocamidesTest[1, "Error"])  # -741.8
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.25), "Error"])  # -99.7
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.5), "Error"])  # 9.8
    print(inputCocamidesTest[floor(Int, size(inputCocamidesTest, 1)*0.75), "Error"])  # 70.2
    print(inputCocamidesTest[size(inputCocamidesTest, 1)*1, "Error"])  # 359.7
    describe(inputCocamidesTest)
    # avg = -18.1

## output
    output = vcat(inputCocamidesTrain, inputCocamidesTest)
    output = output[output.predictRI .!= float(0), :]
    sort!(output, [:dataset, :Error])

savePath = "F:\\UvA\\model1_results.csv"
CSV.write(savePath, output)
