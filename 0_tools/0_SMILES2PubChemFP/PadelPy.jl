using Pkg
Pkg.add("PyCall")
Pkg.add(PackageSpec(url=""))

using CSV, DataFrames
using PyCall
using Conda
Conda.add("padelpy")
Conda.add("pubchempy")
pd = pyimport("padelpy")
pub = pyimport("pubchempy")

#get FPs and missing predicted logP values
# fp_names_Padel = ["PubchemFingerprinter", "SubstructureFingerprintCount", "KlekothaRothFingerprintCount", "
                # AtomPairs2DFingerprintCount", "MACCSFingerprinter"]


function getFPsAndOthers(set)
    # pubchemFPs = "PubchemFP" .* string.(collect(115:262))
    indStart = size(set,2)+1
    indFPs = []
    if indStart > 1000
        println("error all variables provided")
        return set
    end
    for i = 1:size(set,1)
        println(i)

        desc_p = []
        try
            desc_p = DataFrame(pd.from_smiles(set[i,"SMILES"],fingerprints=true, descriptors = false))            #!!! here are the PaDEL fingerprints are calculated
        catch
            println("cannot convert")
            continue
        end
        if isempty(desc_p)
            if i==1
                println("Error at first iteration")
                return
            end
            continue
        end
        #get FPs
        if i == 1
            #expand dataframe
            for f = 1:size(desc_p,2)
            #    if contains(names(desc_p)[f],"APC2") || any(names(desc_p)[f] .== pubchemFPs)
                indFPs = [indFPs;f]
                set[!,names(desc_p)[f]] = fill("",size(set,1))
            #    end
            end
           set[i,indStart:end] = desc_p[1,indFPs]
        else
           set[i,indStart:end] = desc_p[1,indFPs]
        end
    end
    return set
end



#= function getFPsAndOthersAll(set)
    indStart = size(set,2)+1
    if indStart > 20
        println("error all variables provided")
        return set
    end
    for i = 1:size(set,1)
        println(i)
        desc_p = []
        try
            desc_p = DataFrame(pd.from_smiles(set[i,"SMILES"],fingerprints=true, descriptors = false))
        catch
            continue
        end
        if isempty(desc_p)
            if i==1
                println("Error at first iteration")
                return
            end
            continue
        end
        #get FPs
        if i == 1
            #expand dataframe
            for f = 1:size(desc_p,2)
                set[!,names(desc_p)[f]] = fill("",size(set,1))
            end
            set[i,indStart:end] = desc_p[1,:]
        else
            set[i,indStart:end] = desc_p[1,:]
        end
    end
    return set
end =#

###################################################################^
path = "C:/github/exposomeProjUvA/tools/input/dataPubchemFingerprinter.csv"
set = CSV.read(path, DataFrame)
indStart = size(set,2)+1
size(set,1)+1
pd.from_smiles(set[1,"SMILES"],fingerprints=true, descriptors = false)
df = DataFrame(pd.from_smiles(set[1,"SMILES"],fingerprints=true, descriptors = false))
println(df)

#NORMAN
#p1
for i = 1:32
    println(i)
    path = "C:/github/exposomeProjUvA/tools/input/dataPubchemFingerprinter.csv"
    if isfile(path[1:end-4]*"_part_$i.csv")
        continue
    end
    set = CSV.read(path, DataFrame)
    if i == 32
        println(((i-1)*1000)+1:size(set,1))
        set = set[((i-1)*1000)+1:size(set,1),:]
    else
        println(((i-1)*1000)+1:i*1000)
        set = set[((i-1)*1000)+1:i*1000,:]
    end
    set = getFPsAndOthers(set)
    CSV.write(path[1:end-4]*"_part_$i.csv",set)
end


#backward
# for i = 858:-1:1
#     path = "/home/emcms/Data/Denice/CompTox FP/CompTox.csv"
#     set = CSV.read(path, DataFrame)
#     if i == 858
#         println(((i-1)*1000)+1:size(set,1))
#         set = set[((i-1)*1000)+1:size(set,1),:]
#     else
#         println(((i-1)*1000)+1:i*1000)
#         set = set[((i-1)*1000)+1:i*1000,:]
#     end
#     set = getFPsAndOthers(set)
#     CSV.write(path[1:end-4]*"_part_$i.csv",set)
# end


#p2
path = "C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\NORMAN_FP_part2.csv"
set = CSV.read(path, DataFrame)
set
set[:,end-4:end]
set = set[1:10000,1:5]
set = getFPsAndOthers(set)
CSV.write("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\NORMAN_FP_part2-1.csv",set)

#p3
path = "C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\NORMAN_FP_part3.csv"
set = CSV.read(path, DataFrame)
set
set[:,end-4:end]
set = set[16563:end,1:5]        !!!!
set = getFPsAndOthers(set)
CSV.write("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\NORMAN_FP_part3-2.csv",set)




## Add files back together
using CSV, DataFrames
#p1
p1 = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\NORMAN_FP_part1.csv", DataFrame)
p2 = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\NORMAN_FP_part1-1.csv", DataFrame)
p3 = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\NORMAN_FP_part1-2.csv", DataFrame)
p1 = vcat(p1[1:19163,:], p2)
p1 = vcat(p1,p3)
CSV.write("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\NORMAN_FP_part1.csv", p1)





function rerunMissing(p1)

    ind = findall(ismissing.(p1[!,"APC2D10_B_Si"]))
    if isempty(ind)
        println("All done")
        return p1
    end

    set = DataFrame(SMILES = p1[!,"SMILES"][ind])

    pubchemFPs = "PubchemFP" .* string.(collect(115:262))
    indStart = size(set,2)+1
    indFPs = []
    if indStart > 20
        println("error all variables provided")
        return set
    end
    for i = 1:size(set,1)
        println(i)
        # if ismissing(set[i,"SMILES"])
        #     #get smiles from Inchikey
        #     comp = pub.get_compounds(set[i,"INCHIKEY"], "inchikey")
        #     if length(comp) > 1
        #         len = ones(length(comp)).*Inf
        #         for s = 1:length(comp)
        #             len[s] = length(comp[s].canonical_smiles)
        #         end
        #         set[i, "SMILES"] = comp[argmin(len)].canonical_smiles
        #     elseif isempty(comp)
        #         continue
        #     else
        #         set[i, "SMILES"] = comp[1].canonical_smiles
        #     end
        # end
        desc_p = []
        try
            desc_p = DataFrame(pd.from_smiles(set[i,"SMILES"],fingerprints=true, descriptors = false))
        catch
            continue
        end
        if isempty(desc_p)
            if i==1
                println("Error at first iteration")
                return
            end
            continue
        end
        #get FPs
        if i == 1
            #expand dataframe
            for f = 1:size(desc_p,2)
                if contains(names(desc_p)[f],"APC2") || any(names(desc_p)[f] .== pubchemFPs)
                    indFPs = [indFPs;f]
                    set[!,names(desc_p)[f]] = fill("",size(set,1))
                end
            end
            set[i,indStart:end] = desc_p[1,indFPs]
        else
            set[i,indStart:end] = desc_p[1,indFPs]
        end
    end
    #save to temp and load
    CSV.write("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\temp\\tempFP.csv", set)
    set = CSV.read("C:\\Users\\dherwer\\OneDrive - UvA\\Projects\\Alex\\New approach\\temp\\tempFP.csv", DataFrame)
    set = Matrix(set[:,2:end])

    #merge results
    for i = 1:length(ind)
        p1[ind[i],6:end] = set[i,:]
    end

    return p1
end