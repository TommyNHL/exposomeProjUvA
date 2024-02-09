function getVec(matStr)
  if matStr[1] .== '['
      if contains(matStr, ", ")
          str = split(matStr[2:end-1],", ")
      else
          str = split(matStr[2:end-1]," ")
      end
  elseif matStr[1] .== 'A'
      if contains(matStr, ", ")
          str = split(matStr[5:end-1],", ")
      else
          str = split(matStr[5:end-1]," ")
      end
  elseif matStr[1] .== 'F'
      if matStr .== "Float64[]"
          return []
      else
          str = split(matStr[9:end-1],", ")
      end
  elseif matStr[1] .== 'I'
      if matStr .== "Int64[]"
          return []
      else
          str = split(matStr[7:end-1],", ")
      end
  else
      println("New vector start")
      println(matStr)
  end
  if length(str) .== 1 && cmp(str[1],"") .== 0
      return []
  else
      str = parse.(Float64, str)
      return str
  end
end