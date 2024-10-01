# exposomeProjUvA
# Procedure

## 0_tools  (folder)
### A_Installer11.ipynb

## 1_model1  (folder)
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

## 2_model2  (folder)
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

## 3_model3  (folder)
## 3_I_prepareSemisynData  (sub-folder)
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
### D_TPTNmodelTrainTestSplit_ALL.jl
#### - INPUT(S): ***dataframeTPTNModeling_1.csv - dataframeTPTNModeling_8.csv and dataframeTPTNModeling_pest.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_all.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_withLeverage_all.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_TrainIndex_all.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_TrainDFwithhl_all.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_TrainYesIndex_all.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_TrainYesDFwithhl_all.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_TestIndex_all.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_TestDFwithhl_all.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_TestYesIndex_all.csv***
#### - OUTPUT(S): ***dataframeTPTNModeling_TestYesDFwithhl_all.csv***
## 3_II_readyPestData  (sub-folder)
### E_TestSamplesPre.jl
#### - INPUT(S): ***CNL_Ref.csv***
#### - OUTPUT(S): ***CNL_Ref_PestMix_1-8.csv***
#### - OUTPUT(S): ***INCHIKEYs_CNL_Ref_PestMix_1-8.csv***
#### - OUTPUT(S): ***INCHIKEYs_CNL_Ref_PestMix_1.csv - INCHIKEYs_CNL_Ref_PestMix_8.csv***
### F_TestTemplate.jl  (for test samples in the foler- PestMix1-8_test_report_comp_IDs)
#### - INPUT(S): ***INCHIKEYs_CNL_Ref_PestMix_1-8.csv***
#### - INPUT(S): ***PestMix1-8_test_report_comp_IDs.csv***
#### - INPUT(S): ***dataAllFP_withNewPredictedRiWithStratification.csv***
#### - INPUT(S): ***TPTN_dfCNLfeaturesStr.csv***
#### - INPUT(S): ***CocamideExtended73_CNLsRi_RFwithStratification.joblib***
#### - OUTPUT(S): ***PestMix1-8_test_report_comp_IDs_check.csv***
#### - OUTPUT(S): ***PestMix1-8_test_report_comp_IDs_checked.csv***
#### - OUTPUT(S): ***PestMix1-8_test_report_comp_IDs_ready4CNLdf.csv***
#### - OUTPUT(S): ***PestMix1-8_test_report_comp_IDs_extractedWithCNLsList.csv***
#### - OUTPUT(S): ***PestMix1-8_test_report_comp_IDs_withCNLRideltaRi.csv***
#### - OUTPUT(S): ***PestMix1-8_test_report_comp_IDs_dataframeTPTNModeling.csv***
## 3_III_EDA  (sub-folder)
### G_PreEDA.jl
#### - INPUT(S): ***PestMix1-8_1ug-L_NoTea_test_report_comp_IDs_dataframeTPTNModeling.csv***
#### - INPUT(S): ***PestMix1-8_1ug-L_Tea_test_report_comp_IDs_dataframeTPTNModeling***
#### - INPUT(S): ***dataframeTPTNModeling_TrainYesDFwithhl_all.csv***
#### - INPUT(S): ***dataframeTPTNModeling_TestYesDFwithhl_all.csv***
#### - OUTPUT(S): ***allRealsampleNoTea_dataframeTPTNModeling.csv***
#### - OUTPUT(S): ***allRealsampleWithTea_dataframeTPTNModeling.csv***
#### - OUTPUT(S): ***trainDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
#### - OUTPUT(S): ***trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
#### - OUTPUT(S): ***trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - OUTPUT(S): ***testDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
#### - OUTPUT(S): ***testDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
#### - OUTPUT(S): ***testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - OUTPUT(S): ***noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
#### - OUTPUT(S): ***noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
#### - OUTPUT(S): ***noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - OUTPUT(S): ***TeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
#### - OUTPUT(S): ***TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
#### - OUTPUT(S): ***TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
### H_EDA_RefMatchFragRatio.jl
### I_EDA_UsrMatchFragRatio.jl
### J_EDA_MS1Error.jl
### K_EDA_MS2Error.jl
### L_EDA_MS2ErrorStd.jl
### M_EDA_Match.jl
### N_EDA_FinalScoreRatio.jl
### O_EDA_MonoisotopicMass.jl
### P_EDA_DeltaRI.jl
#### - INPUT(S): ***trainDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
#### - INPUT(S): ***trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
#### - INPUT(S): ***trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - INPUT(S): ***testDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
#### - INPUT(S): ***testDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
#### - INPUT(S): ***testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - INPUT(S): ***noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
#### - INPUT(S): ***noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
#### - INPUT(S): ***noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - INPUT(S): ***TeaDF_dataframeTPTNModeling_0d5FinalScoreRatio.csv***
#### - INPUT(S): ***TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDE.csv***
#### - INPUT(S): ***TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - OUTPUT(S): ***outplot_TPTNDistrution_FeatureRefMatchFragRatio_noFilter.png***
#### - OUTPUT(S): ***outplot_TPTNDistrution_FeatureUsrMatchFragRatio_noFilter.png***
#### - OUTPUT(S): ***outplot_TPTNDistrution_FeatureMS1Error_noFilter.png***
#### - OUTPUT(S): ***outplot_TPTNDistrution_FeatureMS2Error_noFilter.png***
#### - OUTPUT(S): ***outplot_TPTNDistrution_FeatureMS2ErrorStd_noFilter.png***
#### - OUTPUT(S): ***outplot_TPTNDistrution_FeatureMatch_noFilter.png***
#### - OUTPUT(S): ***outplot_TPTNDistrution_FeatureFinalScoreRatio_noFilter.png***
#### - OUTPUT(S): ***outplot_TPTNDistrution_FeatureMonoisotopicMass_noFilter.png***
#### - OUTPUT(S): ***outplot_TPTNDistrution_FeatureDeltaRi_noFilter.png***
## 3_IV_featureModelSelection  (sub-folder)
### Q_FeatureSelection.ipynb
#### - INPUT(S): ***trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - INPUT(S): ***testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - INPUT(S): ***noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - INPUT(S): ***TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - OUTPUT(S): ***sns_heatmap(  )***
### R_FeatureModelSelection.jl
#### - INPUT(S): ***trainDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - INPUT(S): ***testDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - INPUT(S): ***noTeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - INPUT(S): ***TeaDF_dataframeTPTNModeling_0d5FinalScoreRatioDEnoFilterSTD.csv***
#### - OUTPUT(S): ***hyperparameterTuning_modelSelection_RF(n)_noFilter_noLog(UsrFragMatchRatio).csv***
#### - OUTPUT(S): ***hyperparameterTuning_modelSelection_DT(n)_noFilter_noLog(UsrFragMatchRatio).csv***
#### - OUTPUT(S): ***hyperparameterTuning_modelSelection_KNN(n)_noLog(UsrFragMatchRatio)_noMatchDiff.csv***
#### - OUTPUT(S): ***hyperparameterTuning_modelSelection_KNN(n)_noLog(UsrFragMatchRatio)_noMatchDiff_noDeltaRI.csv***
#### - OUTPUT(S): ***hyperparameterTuning_modelSelection_LR(n)_noFilter_noLog(UsrFragMatchRatio).csv"***
### T_Plot4ModelFeatureSelection.ipynb
#### - INPUT(S): ***hyperparameterTuning_modelSelection_LR_noFilterSummary.xlsx***
#### - INPUT(S): ***hyperparameterTuning_modelSelection_DT_noFilterSummary.xlsx***
#### - INPUT(S): ***hyperparameterTuning_modelSelection_RF_noFilterSummary.xlsx***
#### - INPUT(S): ***hyperparameterTuning_modelSelection_KNN_noFilterSummary.xlsx***
#### - OUTPUT(S): ***updated4pestNoTea_8paraVS7para.jpg***
#### - OUTPUT(S): ***updated4pest_8paraVS7para.jpg***
#### - OUTPUT(S): ***updatedPestRecall_8paraVS7para.jpg***
## 3_V_modelEvaluation  (sub-folder)
### U_ModelEvaluation.jl
#### - INPUT(S): ***hyperparameterTuning_modelSelection_LR_noFilterSummary.xlsx***
#### - INPUT(S): ***hyperparameterTuning_modelSelection_DT_noFilterSummary.xlsx***
#### - INPUT(S): ***hyperparameterTuning_modelSelection_RF_noFilterSummary.xlsx***
#### - INPUT(S): ***hyperparameterTuning_modelSelection_KNN_noFilterSummary.xlsx***
#### - OUTPUT(S): ***
### V_TPTNmodelCutOff.jl

