in1 = 0

function binning(mzInput)
    setrounding(BigFloat, RoundUp) do 
      in1 = BigFloat(mzInput) + parse(BigFloat, "0.01")
      return in1
    end
end


println(binning(1.1555))

round(binning(1.1555), RoundDown, digits = 2)

round(1.1500000000000, RoundDown, digits = 2)