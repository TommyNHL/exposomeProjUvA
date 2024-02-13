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
### Filtering4NLsCalculation.jl
#### - step 1: filter in positive ionization mode
#### - step 2: filter in precusor ion with measured m/z
#### - step 3: filter in precusor ion with m/z <= 1000
#### - step 4: bin the m/z domain with bin size 0.01 Da (steps)
#### - step 5: calculate NLs by m/z(precusor ion) - m/z(fragment ions)
####           -> new .csv
### JoiningInternalDB.jl
#### - step 6: repeat steps 1-5
#### - step 7: join with Cocamide MS2 spectra DB
### MergingNLs2CNLsMasses.jl
#### - step 8: merge NLs into list for each ID copound
####           -> new .csv
### CNLsFeaturing.jl ***re-run overnight later***
#### - step 9: transform table as row(ID copounds) x column(CNLs masses)
####           -> new .csv
####           -> new .png
### ScoresCNLsCalculation.jl ***workingOn***
#### - step 10: calculate SUM(P(TP)) for each CNL
#### - step 11: calculate SUM(P(TN)) for each CNL
#### - step 12: calculate Score(CNL) for each CNL = 1 - SUM(P(TP)) / SUM(P(TN))
#### - step 13: filter in CNL with Score(CNL) >= 0.0
####           -> new .csv
### MatchingCNLsOfInterest.jl ***pending***
#### - step 14: filter in CNLs features according to the pre-defined CNLs in CNLs_10mDa.csv
### Dfs4CNLmodeling.jl ***pending***
####           -> new .csv
#### - step 15: join the table with the FP-derived Ri values by keys SMILES || INCHIKEY
####           -> new .csv

## 3_trainTestCNLmodel ***pending***