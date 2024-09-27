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
### F_CNLmodelTrainValTest.jl
#### - INPUT(S): ***dataframe73_95dfTestSetWithStratification.csv***
#### - INPUT(S): ***dataframe73_95dfTrainSetWithStratification.csv***
#### - INPUT(S): ***CocamideExtWithStartification_Fingerprints_train.csv***
#### - INPUT(S): ***CocamideExtWithStratification_Fingerprints_test.csv***
#### - INPUT(S): ***dataAllFP_withNewPredictedRiWithStratification.csv***
#### - OUTPUT(S): ***hyperparameterTuning_RFwithStratification10F.csv***
#### - OUTPUT(S): ***CocamideExtended73_CNLsRi_RFwithStratification.joblib***
#### - OUTPUT(S): ***dataframe73_dfTrainSetWithStratification_withCNLPredictedRi.csv***
#### - OUTPUT(S): ***dataframe73_dfTestSetWithStratification_withCNLPredictedRi.csv***
#### - OUTPUT(S): ***dataframe73_dfTrainSetWithStratification_withCNLPredictedRi_withCocamides.csv***
#### - OUTPUT(S): ***dataframe73_dfTestSetWithStratification_withCNLPredictedRi_withCocamides.csv***
#### - OUTPUT(S): ***CNLRiPrediction73_RFTrainWithStratification_v3.png***
#### - OUTPUT(S): ***CNLRiPrediction73_RFTestWithStratification_v3.png***
### G_CNLdfLeverage.jl
#### - INPUT(S): ***dataframe73_95dfTrainSetWithStratification.csv***
#### - INPUT(S): ***dataframe73_dfTrainSetWithStratification_95FPCNLleverage.csv***
#### - OUTPUT(S): ***CNL model 95% leverage cut-off = 0.14604417882015916***
#### - OUTPUT(S): ***CNLLeverageValueDistrution.png***

## IV_model3  (folder)
## IV_I_prepareSemisynData  (sub-folder)
### A_DataSplitMatch.jl
#### - INPUT(S): ***Cand_synth_rr10_1_1000.csv***
#### - INPUT(S): ***Cand_synth_rr10_1001_2000.csv***
#### - INPUT(S): ***Cand_synth_rr10_2001_3000.csv***
#### - INPUT(S): ***Cand_synth_rr10_3001_4000.csv***
#### - INPUT(S): ***Cand_synth_rr10_4001_5000.csv***
#### - INPUT(S): ***dataAllFP_withNewPredictedRiWithStratification.csv***
#### - INPUT(S): ***generated_susp.csv***
#### - OUTPUT(S): ***Cand_synth_rr10_1-5000.csv***
#### - OUTPUT(S): ***Cand_synth_rr10_1-5000_extractedWithoutDeltaRi.csv***
#### - OUTPUT(S): ***Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_trainValDf.csv***
#### - OUTPUT(S): ***Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_isotestDf.csv***
### B_DfCNLdeploy.jl
#### - INPUT(S): ***Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_trainValDf.csv or Cand_synth_rr10_1-5000_extractedWithoutDeltaRi_isotestDf.csv***
#### - INPUT(S): ***dataframe73_dfTestSetWithStratification_withCNLPredictedRi.csv***
#### - INPUT(S): ***CocamideExtended73_CNLsRi_RFwithStratification.joblib***
#### - OUTPUT(S): ***TPTN_dfCNLfeaturesStr.csv***
#### - OUTPUT(S): ***Cand_synth_rr10_1-5000_extractedWithCNLsList.csv or Cand_synth_rr10_1-5000_extractedWithCNLsList_pest.csv***
#### - OUTPUT(S): ***dfCNLsSum_1.csv - dfCNLsSum_8.csv or dfCNLsSum_pest.csv***
#### - OUTPUT(S): ***TPTNmassesCNLsDistrution_8.png***
#### - OUTPUT(S): ***dataframeCNLsRows4TPTNModeling_8withCNLRideltaRi.csv or dataframeCNLsRows4TPTNModeling_PestwithCNLRideltaRi.csv***
#### - OUTPUT(S): ***dataframeCNLsRows4TPTNModeling_TPOnlywithCNLRideltaRi.csv or dataframeCNLsRows4TPTNModeling_TPOnlywithCNLRideltaRi_pest.csv***
#### - OUTPUT(S): ***dfCNLsSum_TP.csv or dfCNLsSum_TP_pest.csv***
#### - OUTPUT(S): ***dfCNLsSum.csv***
#### - OUTPUT(S): ***TPTNmassesCNLsDistrution.png or TPTNmassesCNLsDistrution_pest.png***
### C_TPTNmodelCNLleverageCutoff.jl
#### - INPUT(S): ***dataframeCNLsRows4TPTNModeling_1withCNLRideltaRi.csv - dataframeCNLsRows4TPTNModeling_8withCNLRideltaRi.csv or dataframeCNLsRows4TPTNModeling_PestwithCNLRideltaRi.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_1.csv - dataframeTPTNModeling_8.csv or dataframeTPTNModeling_pest.csv***

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