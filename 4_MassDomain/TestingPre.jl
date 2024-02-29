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

# inputing 693677 x 3+21567 df
# columns: ENTRY, SMILES, INCHIKEY, CNLmasses...
inputDf = CSV.read("F:\\Cand_search_rr0_0612_TEST_100-400_extractedWithDeltaRi.csv", DataFrame)
testIDsDf = CSV.read("F:\\generated_susp.csv", DataFrame)

testIDs = Set()
for i in 1:size(testIDsDf, 1)
    push!(testIDs, testIDsDf[i, "INCHIKEY"])
end
distinctTestIDs = sort!(collect(testIDs))

keepIdx = []
isolateIdx = []
for i in 1:size(inputDf, 1)
    if (inputDf[i, "INCHIKEY"] in testIDs)
        push!(isolateIdx, i)
    else
        push!(keepIdx, i)
    end
end

keepIdx
isolateIdx

trainValDf = inputDf[keepIdx, :]
# save, ouputing testSet df 0.3 x (3+15994+1)
savePath = "F:\\Cand_search_rr0_0612_TEST_100-400_trainValDf.csv"
CSV.write(savePath, trainValDf)

testDf = inputDf[isolateIdx, :]
# save, ouputing testSet df 0.3 x (3+15994+1)
savePath = "F:\\Cand_search_rr0_0612_TEST_100-400_testDf.csv"
CSV.write(savePath, testDf)