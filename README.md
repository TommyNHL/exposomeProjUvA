# exposomeProjUvA
# Procedure

## I_tools  (folder)
### A_Installer11.ipynb
### B_CombinePubchemFP.jl
#### - INPUT(S): ***dataPubchemFingerprinter.csv***
#### - OUTPUT(S): ***dataPubchemFingerprinter_converted.csv***
### C_MergeApc2dPubchem.jl
#### - INPUT(S): ***dataAP2DFingerprinter.csv***
#### - INPUT(S): ***dataPubchemFingerprinter_converted.csv***
#### - OUTPUT(S): ***dataAllFingerprinter_4RiPredict.csv***

## 1_2_3_WithStratification  (folder)
### A_ModelDeploy4FPbasedRi.jl  ***(sub-directory folder "1_deployRFmodel")***
#### - step 01: load the pre-train RF-based model- CocamideExtended.joblib
#### - step 02: predict the FP-derived Ri values
####           -> new .csv
### B_PreProcessInternalDB.jl  ***(sub-directory folder "2_prepareCNLmodel")***
#### - step 03: filter in positive ionization mode
#### - step 04: filter in precusor ion with measured m/z
#### - step 05: bin the m/z domain with bin size 0.01 Da (steps)
#### - step 06: calculate NLs by m/z(precusor ion) - m/z(fragment ions)
#### - step 07: match CNLs-of-interest according to the pre-defined CNLs in CNLs_10mDa.csv
####           -> new .csv
#### - step 08: filter out rows with <2 CNL features
#### - step 09: collect Entry-of-interest according to the presence of FPs in .csv DB
#### - step 10: remove duplicated row spectrum 
####           -> new .csv
#### - step 11: include the pre-ion
#### - step 12: transform table as row(ID copounds) x column(CNLs masses)
####           -> new .csv
####           -> new .png
### C_FPDfPre4Leverage.jl
#### - step 13: find distinct INCHIKEYs
####           -> new .csv
#### - step 14: count frequency for each INCHIKEY
####           -> new .csv
#### - step 15: creat a FP df after taking INCHIKEY frequency into account
####           -> new .csv
### D_trainTestSplitPre.jl
#### - step 16: extract column-of-interests for CNL df construction
####           -> new .csv
####           -> new .csv
####           -> new .csv
####           -> new .csv
### E_LeverageGetIdx.jl
#### - step 17: calculate leverage value
#### - step 18: record leverage value and train/test group information
####           -> new .csv
####           -> new .csv
####           -> new .csv
### F_TrainTestSplit.jl
#### - step 19: perform 5:5 train/test split by index
#### - step 20: gather and join ID informaiton and FP and FP-Ri
####           -> new .csv
####           -> new .csv
### G_CNLmodelTrainVal(Test).jl  ***(sub-directory folder "3_trainTestCNLmodel")***
#### - step 21: split the table for train & test by leverage values
#### - step 22: tune hyper-parameters via CV = 3
#### - step 23: train model
####           -> new .joblib model
#### - step 24: precdict CNL-based Ri values for the internally split cocamides val set
#### - step 25: analyze model predictive power
####           -> new .csv
####           -> new .csv
####           -> new .csv
####           -> new .csv
#### - step 26: match SMILES ID to each INCHIKEY ID
#### - step 27: plot scatter plots for Cocamides & Non-Cocamides via SMILES ID
####           -> new .png
####           -> new .png

## 4_MassDomain  (folder)
### A_DataSplitMatch.jl
#### - step 01: import results from ULSA
#### - step 02: extract useful columns including 7+1 features
#### - step 03: combine different .csv files
#### - step 04: take ratios
#### - step 05: add ground-truth labels based on INCHIKEYreal vs. INCHIKEY
####           -> new .csv
#### - step 06: match FP-derived Ri by INCHIKEY (measured)
#### - step 07: isolate dependent dataset
####           -> new .csv
####           -> new .csv
### B_DfCNLdeploy.jl
#### - step 08: import 15961 CNL features for model deployment
#### - step 09: prepare CNL df for model deployment
####           -> new .csv
#### - step 10: predict CNL-derived Ri values
#### - step 11: calculate delta Ri
####           -> new .csv
#### - step 12: isolate only Label = 1 samples for statistics discussion
####           -> new .csv
####           -> new .png
### C_TPTNmodelTrainValTest.jl
#### - step 13: extract useful columns
####           -> new .csv
#### - step 14: calculate leverage value
#### - step 15: record leverage value for 8:2 train/test split & the group information
####           -> new .csv
####           -> new .csv
####           -> new .csv
#### - step 16: perform 8:2 train/test split
####           -> new .csv
####           -> new .csv
### D_TPTNmodelTrainValTest.jl
#### - step 17: tune hyper-parameters via CV = 3
####           -> new .csv
#### - step 18: train model
####           -> new .joblib model
####           -> new .joblib model
#### - step 19: precdict CNL-based Ri values for the internally split sets
####           -> new .csv
####           -> new .csv
#### - step 20: analyze model predictive power
####           -> new .csv
####           -> new .csv
####           -> new .csv
####           -> new .csv
### E_TPTNmodelzTPRTNR.jl
#### - step 21: calculate statistics for confusion matrix plot
####           -> new .csv
####           -> new .csv
####           -> new .png
####           -> new .png
#### - step 22: plot P(1) threshold-to-TPR/FNR curve
####           -> new .csv
####           -> new .csv
####           -> new .png
#### - step 23: plot P(1) threshold-to-TNR/FPR curve
####           -> new .csv
####           -> new .csv
####           -> new .png
### F_TPTNmodelzzDiscuss.jl
#### - step 24: calculate statistics for delta Ri plot
####           -> new .csv
####           -> new .png