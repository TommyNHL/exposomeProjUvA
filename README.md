# exposomeProjUvA
# Procedure

## I_tools  (folder)
### A_Installer11.ipynb

## II_model1  (folder)
### A_CombinePubchemFP.jl
#### - INPUT(S): ***dataPubchemFingerprinter.csv***
#### - OUTPUT(S): ***dataPubchemFingerprinter_converted.csv***
### B_MergeApc2dPubchem.jl
#### - INPUT(S): ***dataAP2DFingerprinter.csv***
#### - INPUT(S): ***dataPubchemFingerprinter_converted.csv***
#### - OUTPUT(S): ***dataAllFingerprinter_4RiPredict.csv***
### C_ModelDeploy4FPbasedRi.jl
#### - INPUT(S): ***dataAllFingerprinter_4RiPredict.csv***
#### - INPUT(S): ***CocamideExtendedWithStratification.joblib***
#### - OUTPUT(S): ***dataAllFP_withNewPredictedRiWithStratification.csv***

## III_model2  (folder)
## I_prepareCNLmodel  (sub-folder)
### A_PreProcessInternalDB.jl (README_dbColHeaders.md)
#### - INPUT(S): ***Database_INTERNAL_2022-11-17.csv***
#### - INPUT(S): ***CNLs_10mDa.csv***
#### - INPUT(S): ***dataAllFP_withNewPredictedRiWithStratification.csv***
#### - OUTPUT(S): ***CNLs_10mDa_missed.csv***
#### - OUTPUT(S): ***databaseOfInternal_withNLs.csv***
#### - OUTPUT(S): ***dataframeCNLsRows.csv***
#### - OUTPUT(S): ***dfCNLsSumModeling.csv***
#### - OUTPUT(S): ***massesCNLsDistrution.png***
### B_FPDfPre4Leverage.jl
#### - INPUT(S): ***databaseOfInternal_withNLs.csv***
#### - INPUT(S): ***dataframeCNLsRows.csv***
#### - INPUT(S): ***dataAllFP_withNewPredictedRiWithStratification.csv***
#### - OUTPUT(S): ***countingRows4Leverage.csv***
#### - OUTPUT(S): ***countingRowsInFP4Leverage.csv***
#### - OUTPUT(S): ***dataAllFP_withNewPredictedRiWithStratification_Freq.csv***
### C_TrainTestSplitPre.jl
#### - INPUT(S): ***dataframeCNLsRows.csv***
#### - INPUT(S): ***dataAllFP_withNewPredictedRiWithStratification_Freq.csv***
#### - OUTPUT(S): ***databaseOfInternal_withNLsOnly.csv***
#### - OUTPUT(S): ***databaseOfInternal_withEntryInfoOnly.csv***
#### - OUTPUT(S): ***databaseOfInternal_withINCHIKEYInfoOnly.csv***
#### - OUTPUT(S): ***databaseOfInternal_withYOnly.csv***
### D_LeverageGetIdx.jl
#### - INPUT(S): ***dataAllFP_withNewPredictedRiWithStratification_Freq.csv***
#### - OUTPUT(S): ***dataframe73_dfTrainSetWithStratification_95index.csv***
#### - OUTPUT(S): ***dataframe73_dfTestSetWithStratification_95index.csv***
#### - OUTPUT(S): ***dataframe73_dfWithStratification_95index.csv***
#### - OUTPUT(S): ***dataAllFP73_withNewPredictedRiWithStratification_FreqAnd95Leverage.csv***
### E_TrainTestSplit.jl
#### - INPUT(S): ***databaseOfInternal_withEntryInfoOnly.csv***
#### - INPUT(S): ***databaseOfInternal_withINCHIKEYInfoOnly.csv***
#### - INPUT(S): ***databaseOfInternal_withNLsOnly.csv***
#### - INPUT(S): ***databaseOfInternal_withYOnly.csv***
#### - INPUT(S): ***dataframe73_dfTrainSetWithStratification_95index.csv***
#### - INPUT(S): ***dataframe73_dfTestSetWithStratification_95index.csv***
#### - OUTPUT(S): ***dataframe73_95dfTrainSetWithStratification.csv***
#### - OUTPUT(S): ***dataframe73_95dfTestSetWithStratification.csv***

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