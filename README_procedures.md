# exposomeProjUvA

# Procedure
## 0_tools (folder)
### CombinePubchemFP.jl in the sub-directory 1_CombinePubChemFP
# - step 1: convert 148 Pubchem FPs features into 10 condensed FPs feastures 
#           -> new .csv
### MergeAp2dPubchem.jl in the sub-directory 2_MergeAp2dPubchem 
# - step 2: join 780 APC2D FPs features table with the 10 Pubchem-based condensed FPs festures table
#           -> new .csv

## 1_deployRFmodel (folder)
### ModelDeploy4FPbasedRi.jl
# - step 1: load the pre-train RF-based model- CocamideExtended.joblib
# - step 2: predict the FP-derived Ri values
#           -> new .csv
### FPbasedRiADCalculation.jl
# ***to be continued***

## 2_prepareCNLmodel (folder)
### Filtering4NLsCalculation.jl
# - step 1: filter in positive ionization mode
# - step 2: filter in precusor ion with measured m/z
# - step 3: filter in precusor ion with m/z <= 1000
# - step 4: bin the m/z domain with bin size 0.01 Da (steps)
# - step 5: calculate NLs by m/z(precusor ion) - m/z(fragment ions)
#           -> new .csv
### MergingNLs2CNLsMasses.jl
# - step 6: merge NLs into list for each ID copound
#           -> new .csv
# - step 7: transform table as row(ID copounds) x column(CNLs masses)
#           -> new .csv
### Df4CNLmodeling.jl
# - step 8: join the table with the FP-derived Ri values by keys SMILES || INCHIKEY
#           -> new .csv

## 3_trainTestCNLmodel