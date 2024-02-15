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
#### - step 5: repeat steps 1-4 ***done***
#### - step 6: join with Cocamide MS2 spectra DB ***ignored***
### MergingNLs2CNLsMasses.jl  ***wasted***
#### - step 7: merge NLs into list for each ID copound
####           -> new .csv
### CNLsFeaturing.jl ***wasted***
#### - step 8: match CNLs-of-interest according to the pre-defined CNLs in CNLs_10mDa.csv
### CNLsFeaturingCopy.jl ***working***
#### - step 8: match CNLs-of-interest according to the pre-defined CNLs in CNLs_10mDa.csv
####           -> new .csv
#### - step 9: transform table as row(ID copounds) x column(CNLs masses)
####           -> new .csv
####           -> new .png
### ScoresCNLsCalculation.jl ***wasted***
#### - step --: calculate SUM(P(TP)) for each CNL
#### - step --: calculate SUM(P(TN)) for each CNL
####           -> new .csv
#### - step --: calculate Score(CNL) for each CNL = 1 - SUM(P(TP)) / SUM(P(TN))
####           -> new .csv
### Dfs4CNLmodeling.jl ***re-run overnight***
#### - step 10: split the table with only the cocamides
#### - step 11: merge the table with only the cocamides with the FP-based Ri
####           -> new .csv
#### - step 12: split the table without the cocamides
####           -> new .csv

## 3_trainTestCNLmodel
### CNLmodelTrainVal.jl ***working***
#### - step 1: split the table with only the cocamides for train & test
####           -> new .csv
####           -> new .csv
#### - step 2: tune hyper-parameters
#### - step 3: train model
#### - step 4: precdict CNL-derived Ri values for the table without the cocamides 
