
function testApproxMPS()

  D = 20
  pd = 6

  Big = [zeros(D,pd,D) for j = 1:N]
  Big[1] = zeros(1,pd,D)
  Big[N] = zeros(D,pd,1)

  for j = 1:N
    if !iseven(j)
      for d = 1:pd
        Big[j][1,d,d]=1
        Big[j+1][d,d,1]=1
      end
    end
  end

  approxMPS(Big,4)

end
