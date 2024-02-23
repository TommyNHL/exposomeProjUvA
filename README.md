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
### ModelDeploy4FPbasedRi.jl  ***done***
#### - step 1: load the pre-train RF-based model- CocamideExtended.joblib
#### - step 2: predict the FP-derived Ri values
####           -> new .csv
### DfPre4FPs.jl  ***done***
#### - step 1: find distinct INCHIKEYs
####           -> new .csv
####           -> new .csv
#### - step 2: count frequency for each INCHIKEY
####           -> new .csv
#### - step 3: creat a FP df after taking INCHIKEY frequency into account
####           -> new .csv
### TrainTestSplitPre.jl  ***done***
#### - step 4: extract column-of-interests for CNL df construction
####           -> new .csv
####           -> new .csv
####           -> new .csv
####           -> new .csv
### LeverageGetIdx.jl  ***done***
#### - step 5: calculate leverage value
#### - step 6: record leverage value and train/test group information
####           -> new .csv
####           -> new .csv
####           -> new .csv
### TrainTestSplit.jl  ***done***
#### - step 7: perform 7:3 train/test split by index
#### - step 8: gather and join ID informaiton and FP and FP-Ri
####           -> new .csv
####           -> new .csv

## 2_prepareCNLmodel (folder)
### PreProcessInternalDB.jl  ***done***
#### - step 1: filter in positive ionization mode
#### - step 2: filter in precusor ion with measured m/z
#### - step 3: bin the m/z domain with bin size 0.01 Da (steps)
#### - step 4: calculate NLs by m/z(precusor ion) - m/z(fragment ions)
#### - step 6: match CNLs-of-interest according to the pre-defined CNLs in CNLs_10mDa.csv
#### - step 7: filter out rows with <2 CNL features
#### - step 8: collect Entry-of-interest according to the presence of FPs in .csv DB
#### - step 9: remove duplicated row spectrum 
####           -> new .csv
#### - step 10: include the pre-ion
#### - step 11: transform table as row(ID copounds) x column(CNLs masses)
####           -> new .csv
####           -> new .png
### Dfs4CNLmodeling.jl  ***old version***
#### - step 12: split the table with only the cocamides
#### - step 13: merge the table with only the cocamides with the FP-based Ri
####           -> new .csv
#### - step 14: split the table without the cocamides
####           -> new .csv

## 3_trainTestCNLmodel
### CNLmodelTrainVal(Test).jl  ***pending to run***
#### - step 1: split the table with only the cocamides for train & test  ***old version***
#### - step 1: split the table for train & test by leverage values  ***new version***
####           -> new .csv
####           -> new .csv
#### - step 2: tune hyper-parameters via CV = 3
#### - step 3: train model
#### - step 4: precdict CNL-based Ri values for the internally split cocamides val set
#### - step 5: analyze model predictive power
####           -> new .csv
####           -> new .csv
####           -> new performance metrics
####           -> new performance metrics
#### - step 6: match SMILES ID to each INCHIKEY ID
#### - step 7: plot scatter plots for Cocamides & Non-Cocamides via SMILES ID
####           -> new .png
####           -> new .png
### CNLmodelTest.jl  ***old version***
#### - step 1: load the pre-train CNL-based model- CocamideExtended_CNLsRi.joblib
#### - step 2: precdict CNL-based Ri values for the non-cocamides test set
####           -> new .csv
####           -> new performance metrics
####           -> new .png

## 4_MassDomain
### DataSplitMatch.jl  ***pending to run***
#### - step 1: import results from ULSA
#### - step 2: extract useful columns including 7+1 features
#### - step 3: combine different .csv files
#### - step 4: add ground-truth labels based on INCHIKEYreal vs. INCHIKEY
####           -> new .csv
#### - step 5: calculate and join delta Ri
####           -> new .csv
### TNRemoval.jl  ***skipped***
#### - step 6: remove a portion of rows of TN by leverage values
####           -> new .csv
### TPTNmodelTrainValTest.jl  ***pending to run***
#### - step 7: split the table for train & test by leverage values ***new version***
#### - step 8: tune hyper-parameters via CV = 3
#### - step 9: train model
#### - step 10: precdict CNL-based Ri values for the internally split cocamides val set
#### - step 11: analyze model predictive power
####           -> new .csv
####           -> new .csv
####           -> new performance metrics
####           -> new performance metrics
#### - step 12: match SMILES ID to each INCHIKEY ID
#### - step 13: plot scatter plots for Cocamides & Non-Cocamides via SMILES ID
####           -> new .png
####           -> new .png