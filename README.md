# exposomeProjUvA

# Procedure
## 0_tools (folder)
### CombinePubchemFP.jl in the sub-directory 1_CombinePubChemFP
#### - step 1: convert 148 Pubchem FPs features into 10 condensed FPs feastures 
####           -> new .csv
### MergeAp2dPubchem.jl in the sub-directory 2_MergeAp2dPubchem 
#### - step 2: join 780 APC2D FPs features table with the 10 Pubchem-based condensed FPs festures table
####           -> new .csv

## 1_deployRFmodel (folder)
### ModelDeploy4FPbasedRi.jl
#### - step 1: load the pre-train RF-based model- CocamideExtended.joblib
#### - step 2: predict the FP-derived Ri values
####           -> new .csv
### FPbasedRiADCalculation.jl
#### ***to be continued***

## 2_prepareCNLmodel (folder)
### Filtering4NLsCalculation.jl ***wasted***
#### - step 1: filter in positive ionization mode
#### - step 2: filter in precusor ion with measured m/z
#### - step 3: bin the m/z domain with bin size 0.01 Da (steps)
#### - step 4: calculate NLs by m/z(precusor ion) - m/z(fragment ions)
####           -> new .csv
### JoiningInternalDB.jl
#### - step --: repeat steps 1-4 ***done***
#### - step --: join with Cocamide MS2 spectra DB ***ignored***
### MergingNLs2CNLsMasses.jl  ***wasted***
#### - step --: merge NLs into list for each ID copound
####           -> new .csv
### CNLsFeaturing.jl ***wasted***
#### - step --: match CNLs-of-interest according to the pre-defined CNLs in CNLs_10mDa.csv
### CNLsFeaturingCopy.jl ***done***
#### - step 5: match CNLs-of-interest according to the pre-defined CNLs in CNLs_10mDa.csv
####           -> new .csv
#### - step 6: collect Entry-of-interest according to the presence of FPs in .csv DB
#### - step 7: transform table as row(ID copounds) x column(CNLs masses)
####           -> new .csv
####           -> new .png
### ScoresCNLsCalculation.jl ***wasted***
#### - step --: calculate SUM(P(TP)) for each CNL
#### - step --: calculate SUM(P(TN)) for each CNL
####           -> new .csv
#### - step --: calculate Score(CNL) for each CNL = 1 - SUM(P(TP)) / SUM(P(TN))
####           -> new .csv
### Dfs4CNLmodeling.jl ***done***
#### - step 8: split the table with only the cocamides
#### - step 9: merge the table with only the cocamides with the FP-based Ri
####           -> new .csv
#### - step 10: split the table without the cocamides
####           -> new .csv

## 3_trainTestCNLmodel
### CNLmodelTrainVal.jl ***done***
#### - step 1: split the table with only the cocamides for train & test
####           -> new .csv
####           -> new .csv
#### - step 2: tune hyper-parameters
#### - step 3: train model
#### - step 4: precdict CNL-based Ri values for the internally split cocamides val set
#### - step 5: analyze model predictive power
####           -> new performance metrics
####           -> new .png
### CNLmodelTest.jl ***done***
#### - step 6: load the pre-train CNL-based model- CocamideExtended_CNLsRi.joblib
#### - step 7: precdict CNL-based Ri values for the non-cocamides test set
####           -> new performance metrics
####           -> new .png