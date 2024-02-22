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
### ModelDeploy4FPbasedRi.jl  ***re-predict***
#### - step 1: load the pre-train RF-based model- CocamideExtended.joblib
#### - step 2: predict the FP-derived Ri values
####           -> new .csv
### TrainTestSplit.jl  ***running***
#### - step 1: find distinct INCHIKEYs
####           -> new .csv
####           -> new .csv
#### - step 2: count frequency for each INCHIKEY
####           -> new .csv
#### - step 3: creat a FP df after taking INCHIKEY frequency into account
####           -> new .csv
#### - step 4: calculate leverage value
####           -> new .csv
#### - step 5: perform 7:3 train/test split by index
#### - step 6: gather and join ID informaiton and FP and FP-Ri
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
### Dfs4CNLmodeling.jl ***old version***
#### - step 12: split the table with only the cocamides
#### - step 13: merge the table with only the cocamides with the FP-based Ri
####           -> new .csv
#### - step 14: split the table without the cocamides
####           -> new .csv

## 3_trainTestCNLmodel
### CNLmodelTrainVal(Test).jl ***pending to run***
#### - step 1: split the table with only the cocamides for train & test ***old version***
#### - step 1: split the table for train & test by leverage values ***new version***
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
####           -> new .png
####           -> new .png
#### - step 6: plot scatter plots for Cocamides & Non-Cocamides
### CNLmodelTest.jl ***old version***
#### - step 1: load the pre-train CNL-based model- CocamideExtended_CNLsRi.joblib
#### - step 2: precdict CNL-based Ri values for the non-cocamides test set
####           -> new .csv
####           -> new performance metrics
####           -> new .png

# set min. no. of CNL (3, pre-ion inclusive) ***v***
# include pre-ion ***v***
# perform stratification(mixing) for the FP data, not CNL ***v***
# CV=3 for modeling ***v***
# try CatBoost

## 4_MassDomain
### DataSplitMatch.jl ***working***
#### - step 1: import results from ULSA
#### - step 2: extract useful columns including 7+1 features
#### - step 3: combine different .csv files
####           -> new .csv
#### - step 4: add ground-truth labels based on INCHIKEYreal vs. INCHIKEY
#### - step 5: calculate and join delta Ri
####           -> new .csv
### TPTNmodelTrainValTest.jl ***working***
#### - step 6: split the table for train & test by leverage values ***new version***
#### - step 7: tune hyper-parameters via CV = 3
#### - step 8: train model
#### - step 9: precdict CNL-based Ri values for the internally split cocamides val set
#### - step 10: analyze model predictive power
####           -> new .csv
####           -> new .csv
####           -> new performance metrics
####           -> new performance metrics
####           -> new .png
####           -> new .png
#### - step 11: plot scatter plots for Cocamides & Non-Cocamides