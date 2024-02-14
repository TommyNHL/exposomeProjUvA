## import packages ##
using ScikitLearn
using BSON
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using CSV, DataFrames, PyCall, Conda, Plots, LinearAlgebra, Statistics

@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
jl = pyimport("joblib")



function convertPubChemFPs(ACfp::DataFrame, PCfp::DataFrame)
    FP1tr = ACfp
    pubinfo = Matrix(PCfp)

    #ring counts
    FP1tr[!,"PCFP-r3"] = pubinfo[:,1]
    FP1tr[!,"PCFP-r3"][pubinfo[:,8] .== 1] .= 2
    FP1tr[!,"PCFP-r4"] = pubinfo[:,15]
    FP1tr[!,"PCFP-r4"][pubinfo[:,22] .== 1] .= 2
    FP1tr[!,"PCFP-r5"] = pubinfo[:,29]
    FP1tr[!,"PCFP-r5"][pubinfo[:,36] .== 1] .= 2
    FP1tr[!,"PCFP-r5"][pubinfo[:,43] .== 1] .= 3
    FP1tr[!,"PCFP-r5"][pubinfo[:,50] .== 1] .= 4
    FP1tr[!,"PCFP-r5"][pubinfo[:,57] .== 1] .= 5

    FP1tr[!,"PCFP-r6"] = pubinfo[:,64]
    FP1tr[!,"PCFP-r6"][pubinfo[:,71] .== 1] .= 2
    FP1tr[!,"PCFP-r6"][pubinfo[:,78] .== 1] .= 3
    FP1tr[!,"PCFP-r6"][pubinfo[:,85] .== 1] .= 4
    FP1tr[!,"PCFP-r6"][pubinfo[:,92] .== 1] .= 5
    FP1tr[!,"PCFP-r7"] = pubinfo[:,99]
    FP1tr[!,"PCFP-r7"][pubinfo[:,106] .== 1] .= 2
    FP1tr[!,"PCFP-r8"] = pubinfo[:,113]
    FP1tr[!,"PCFP-r8"][pubinfo[:,120] .== 1] .= 2
    FP1tr[!,"PCFP-r9"] = pubinfo[:,127]
    FP1tr[!,"PCFP-r10"] = pubinfo[:,134]

    #minimum number of type of rings
    arom = zeros(size(pubinfo,1))
    arom[(arom .== 0) .& (pubinfo[:,147] .== 1)] .= 4
    arom[(arom .== 0) .& (pubinfo[:,145] .== 1)] .= 3
    arom[(arom .== 0) .& (pubinfo[:,143] .== 1)] .= 2
    arom[(arom .== 0) .& (pubinfo[:,141] .== 1)] .= 1
    FP1tr[!,"minAromCount"] = arom
    het = zeros(size(pubinfo,1))
    het[(het .== 0) .& (pubinfo[:,148] .== 1)] .= 4
    het[(het .== 0) .& (pubinfo[:,146] .== 1)] .= 3
    het[(het .== 0) .& (pubinfo[:,144] .== 1)] .= 2
    het[(het .== 0) .& (pubinfo[:,142] .== 1)] .= 1
    FP1tr[!,"minHetrCount"] = het

    return FP1tr
end


#load all data
amtr = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\Amide_Fingerprints_train.csv",DataFrame)
amte = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\Amide_Fingerprints_test.csv",DataFrame)
FPatr = Matrix(convertPubChemFPs(amtr[:,4:4+779], amtr[:,4+780:end]))
FPate = Matrix(convertPubChemFPs(amte[:,4:4+779], amte[:,4+780:end]))

uoatr = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\UoA_Fingerprints_train.csv",DataFrame)
uoate = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\UoA_Fingerprints_test.csv",DataFrame)
FPutr = Matrix(convertPubChemFPs(uoatr[:,4:4+779], uoatr[:,4+780:end]))
FPute = Matrix(convertPubChemFPs(uoate[:,4:4+779], uoate[:,4+780:end]))

coctr = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\Cocamide_Fingerprints_train.csv",DataFrame)
cocte = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\Cocamide_Fingerprints_test.csv",DataFrame)
FPctr = Matrix(convertPubChemFPs(coctr[:,9:9+779], coctr[:,9+780:end]))
FPcte = Matrix(convertPubChemFPs(cocte[:,9:9+779], cocte[:,9+780:end]))


## Plot of Series vs series
coctrte = vcat(coctr,cocte)
FPcoc = [FPctr;FPcte]
uoatrte = vcat(uoatr,uoate)
FPuoa = [FPutr;FPute]
amtrte = vcat(amtr,amte)
FPam = [FPatr;FPate]
ca = 0
indaa = []
indac = []
cu = 0
induu = []
induc = []
for i = 1:size(coctrte,1)
    if sum(coctrte[i,"SMILES"] .== uoatrte[!,"SMILES"]) > 0
        cu += sum(coctrte[i,"SMILES"] .== uoatrte[!,"SMILES"])
        induu = [induu;findfirst(coctrte[i,"SMILES"] .== uoatrte[!,"SMILES"])]
        induc = [induc;i]
    end
    if sum(amtrte[:,"SMILES"] .== coctrte[i,"SMILES"]) > 0
        ca += sum(amtrte[:,"SMILES"] .== coctrte[i,"SMILES"])
        indaa = [indaa;findfirst(coctrte[i,"SMILES"] .== amtrte[!,"SMILES"])]
        indac = [indac;i]
    end
end
#get RI links between cocamide with UoA and amide
cocUoA = DataFrame(SMILES = coctrte[induc,"SMILES"], RIcoc = coctrte[induc,"RI"], RIuoa = uoatrte[induu,"RI"])
cocAmide = DataFrame(SMILES = coctrte[indac,"SMILES"], RIcoc = coctrte[indac,"RI"], RIam = amtrte[indaa,"RI"])



## calibration curves
#UoA to coc
x = [ones(size(cocUoA,1),1) cocUoA[:,3]]
y = cocUoA[:,2]
b = pinv(x'*x)*x'*y # Regressed parameters
y_hat = x*b

scatter(cocUoA[:,3], y_hat, label = "Predicted", c = 1)
plot!(cocUoA[:,3], y_hat, label = "", c = 1)
scatter!(cocUoA[:,3], cocUoA[:,2], label = "Measured", c = 2)

xuoa = [ones(size(uoatrte,1),1) uoatrte[!,"RI"]]
cocUoARI = xuoa*b

#Amide to coc
xa = [ones(size(cocAmide,1),1) cocAmide[:,3]]
ya = cocAmide[:,2]
ba = pinv(xa'*xa)*xa'*ya # Regressed parameters
y_hata = xa*ba

scatter(cocAmide[:,3], y_hata, label = "Predicted", c = 1, xlabel = "Measured amide RI", ylabel = "Mapped cocamide RI")
plot!(cocAmide[:,3], y_hata, label = "", c = 1)
scatter!(cocAmide[:,3], cocAmide[:,2], label = "Measured", c = 2)

xamide = [ones(size(amtrte,1),1) amtrte[!,"RI"]]
cocAmRI = xamide*ba


##RF model
allf = vcat(coctr[:,9:end],uoatrte[:,4:end])
allf = vcat(allf,amtrte[:,4:end])
RIs = [coctr[!,"RI"];cocUoARI;cocAmRI]



min_num_leaf_opt = 8
num_tree_opt = 200

state = 1
MaxFeat = Int64(ceil(size(allf,2)/3))
## Regression ##
reg = RandomForestRegressor(n_estimators= num_tree_opt, min_samples_leaf=min_num_leaf_opt, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=state)
fit!(reg, Matrix(convertPubChemFPs(allf[:,1:1+779], allf[:,1+780:end])), RIs)
score(reg, Matrix(convertPubChemFPs(allf[:,1:1+779], allf[:,1+780:end])), RIs)

#test on cocamide test
score(reg, Matrix(convertPubChemFPs(cocte[:,9:9+779], cocte[:,9+780:end])), cocte[!,"RI"])

jl.dump(reg,"C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\CocamideExtended.joblib",compress=5)
CSV.write("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\CocamideExtended_train.csv", allf)



levtr = leverage(Matrix(convertPubChemFPs(all[:,1:1+779], all[:,1+780:end])), Matrix(convertPubChemFPs(all[:,1:1+779], all[:,1+780:end])))
levte = leverage(Matrix(convertPubChemFPs(all[:,1:1+779], all[:,1+780:end])), Matrix(convertPubChemFPs(cocte[:,9:9+779], cocte[:,9+780:end])))

diff = 0.001
bins = collect(0:diff:1)
counts = zeros(length(bins)-1)
for i = 1:length(bins)-1
    counts[i] = sum(bins[i] .<= levtr .< bins[i+1])
end
limthr = bins[findfirst(cumsum(counts)./sum(counts) .> 0.95)-1]

# plot(layout = grid(1,2), size = (800,400), bottommargin = 5mm, leftmargin = 5mm)
sn = "cocamideMap"
scale = 1
tc = 2
RItr = RIs
RIp1 = predict(reg, Matrix(convertPubChemFPs(all[:,1:1+779], all[:,1+780:end])))
scatter([600], [600], markershape = :star, c = :white, label = "Outside AD", dpi =1200)
scatter!(RItr[levtr.<= limthr], RIp1[levtr.<= limthr], xlabel = "Measured "*sn*" RI", ylabel = "Predicted "*sn*" RI", label = "Training set", c = scale)
scatter!(RItr[levtr.> limthr], RIp1[levtr.> limthr], xlabel = "Measured "*sn*" RI", markershape = :star, label = "", c = scale)
RIte = cocte[!,"RI"]
RIp2 = predict(reg, Matrix(convertPubChemFPs(cocte[:,9:9+779], cocte[:,9+780:end])))
scatter!(RIte[levte.<= limthr], RIp2[levte.<= limthr], xlabel = "Measured "*sn*" RI", label = "Testing set", c = tc)
scatter!(RIte[levte.> limthr], RIp2[levte.> limthr], xlabel = "Measured "*sn*" RI", markershape = :star, label = "", c = tc)

mir = minimum([minimum(RItr), minimum(RIte), minimum(RIp1), minimum(RIp2)])
mr = maximum([maximum(RItr), maximum(RIte), maximum(RIp1), maximum(RIp2)])
plot!([mir,mr], [mir,mr], c = :black, label = "1:1 line")



